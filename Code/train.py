from PIL import Image
import os
from copy import deepcopy
from collections import OrderedDict
import math
import random
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from diffusers.models import AutoencoderKL
from model import DiT_B_2
from utils import create_targets
from inference import inference

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

LEARNING_RATE = 0.0001
BATCH_SIZE = 64
EVAL_SIZE = 8

NUM_CLASSES = 5
CLASS_DROPOUT_PROB = 1.0

N_EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('medium')  # or 'high'

LOG_EVERY = 1
CFG_SCALE = 0.0


class CelebaHQDataset(Dataset):
    def __init__(self, im_path, im_size=256, im_channels=3, im_ext='jpg', condition_types=None):
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.im_path = im_path
        self.latent_maps = None
        self.use_latents = False

        self.condition_types = [] if condition_types is None else condition_types

        self.idx_to_cls_map = {}
        self.cls_to_idx_map = {}

        self.images, self.texts = self.load_images(im_path)

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        texts = []

        for fname in tqdm(fnames):
            ims.append(fname)

            if 'text' in self.condition_types:
                im_name = os.path.split(fname)[1].split('.')[0]
                captions_im = []
                with open(os.path.join(im_path, 'celeba-caption/{}.txt'.format(im_name))) as f:
                    for line in f.readlines():
                        captions_im.append(line.strip())
                texts.append(captions_im)

        if 'text' in self.condition_types:
            assert len(texts) == len(ims), "Condition Type Text but could not find captions for all images"

        print('Found {} images'.format(len(ims)))
        print('Found {} captions'.format(len(texts)))
        return ims, texts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}
        if 'text' in self.condition_types:
            cond_inputs['text'] = random.sample(self.texts[index], k=1)[0]
        #######################################
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.im_size),
            torchvision.transforms.CenterCrop(self.im_size),
            torchvision.transforms.ToTensor(), ])(im)
        im.close()

        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        if len(self.condition_types) == 0:
            return im_tensor
        else:
            return im_tensor, cond_inputs


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def train_epoch(train_dataloader, dit, ema, vae, optimizer):

    loss_fn = torch.nn.MSELoss()
    
    dit.train()
    
    total_loss = 0.0
    for batch, (images, texts) in enumerate(tqdm(train_dataloader)):

        images, texts = images.to(DEVICE), texts['text']

        with torch.no_grad():

            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
            
        print(f"latents.shape: {latents.shape}")

        x_t, v_t, t, dt_base= create_targets(latents, texts, dit)
        v_prime = dit(x_t, t, dt_base, texts)

        loss = loss_fn(v_prime, v_t)

        total_loss += loss.item()
        optimizer.zero_grad()
        
        loss.backward()

        optimizer.step()

        update_ema(ema, dit)

    return total_loss / len(train_dataloader)

def evaluate(dit, vae, val_dataloader, epoch):

    dit.eval()

    images, texts_real = next(iter(val_dataloader))
    images, texts_real = images[:EVAL_SIZE].to(DEVICE), texts_real['text'][:EVAL_SIZE]
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor

    fid = FrechetInceptionDistance().to(DEVICE)
    
    images = 255 * ((images - torch.min(images)) / (torch.max(images) - torch.min(images) + 1e-8))

    # all start from same noise
    eps = torch.randn_like(latents).to(DEVICE)

    denoise_timesteps_list = [1, 2, 4, 8, 16, 32, 128]
    # run at varying timesteps
    for i, denoise_timesteps in enumerate(denoise_timesteps_list):
        all_x = []
        delta_t = 1.0 / denoise_timesteps # i.e. step size
        fid.reset()

        x = eps

        for ti in range(denoise_timesteps):
            # t should in range [0,1]
            t = ti / denoise_timesteps

            t_vector = torch.full((eps.shape[0],), t).to(DEVICE)
            dt_base = torch.ones_like(t_vector).to(DEVICE) * math.log2(denoise_timesteps)

            with torch.no_grad():
                v = dit.forward(x, t_vector, dt_base, texts_real)

            x = x + v * delta_t

            if denoise_timesteps <= 8 or ti % (denoise_timesteps//8) == 0 or ti == denoise_timesteps-1:

                with torch.no_grad():
                    decoded = vae.decode(x/vae.config.scaling_factor)[0]
                
                decoded = decoded.to("cpu")

                all_x.append(decoded)
    

        if(len(all_x)==9):
            all_x = all_x[1:]

        # estimate FID metric
        decoded_denormalized = 255 * ((decoded - torch.min(decoded)) / (torch.max(decoded)-torch.min(decoded)+1e-8))
        
        # generated images
        fid.update(images.to(torch.uint8), real=True)
        fid.update(decoded_denormalized.to(torch.uint8).to(DEVICE), real=False)
        fid_val = fid.compute()
        print(f"denoise_timesteps: {denoise_timesteps} | fid_val: {fid_val}")
        with open(r'FID.txt', 'a') as f:
            f.write(f"denoise_timesteps: {denoise_timesteps} | fid_val: {fid_val}\n")
        all_x = torch.stack(all_x)

        def process_img(img):
            # normalize in range [0,1]
            img = img * 0.5 + 0.5
            img = torch.clip(img, 0, 1)
            img = img.permute(1,2,0)
            return img
    
        fig, axs = plt.subplots(8, 8, figsize=(30,30))
        for t in range(min(8, all_x.shape[0])):
            for j in range(8):        
                axs[t, j].imshow(process_img(all_x[t, j]), vmin=0, vmax=1)
        
        fig.savefig(f"epoch_{epoch}_denoise_timesteps_{denoise_timesteps}.png")
        
        plt.close()

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# single GPU, single epoch takes up to 30 min
def main():

    ds = CelebaHQDataset(im_path='data/CelebAMask-HQ',
                        im_size=256,
                        im_channels=3,
                        condition_types=['text'])

    train_dataset, val_dataset = torch.utils.data.random_split(ds, [0.9, 0.1])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True, num_workers=2)

    images_i, _ = next(iter(train_dataloader))

    dit = DiT_B_2(learn_sigma=False,
                  training_type="shortcut").to(DEVICE)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae = vae.eval()
    vae.requires_grad_(False)

    print(f"count_parameters(dit): {count_parameters(dit)}")
    ema = deepcopy(dit).to(DEVICE)
    ema.requires_grad_(False)
    
    # checkpoint_path = "dit_saved72.pth"
    # dit.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.AdamW(dit.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

    update_ema(ema, dit, decay=0)
    ema.eval()

    for i in range(N_EPOCHS):
        print(f"epoch #{i}")
        epoch_loss = train_epoch(train_dataloader, dit, ema, vae, optimizer)
        print(f"epoch_loss: {epoch_loss}")

        with open(r'FID.txt','a') as f:
                f.write(f"{i}: epoch_loss: {epoch_loss}\n")
        evaluate(dit, vae, val_dataloader, i)

        torch.save(dit.state_dict(), f"dit_saved{i}.pth") # save checkpoint

def main_inference():
    ds = CelebaHQDataset(im_path='data/CelebAMask-HQ',
                         im_size=256,
                         im_channels=3,
                         condition_types=['text'])

    train_dataset, val_dataset = torch.utils.data.random_split(ds, [0.9, 0.1])
    # good option is 2*num_gpus
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, drop_last=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, drop_last=True, num_workers=2)

    images_i, _ = next(iter(train_dataloader))

    dit = DiT_B_2(learn_sigma=False,
                  training_type="shortcut").to(DEVICE)

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(DEVICE)
    vae = vae.eval()
    vae.requires_grad_(False)

    print(f"count_parameters(dit): {count_parameters(dit)}")
    ema = deepcopy(dit).to(DEVICE)
    ema.requires_grad_(False)

    update_ema(ema, dit, decay=0)
    ema.eval()
    checkpoint_path = "dit_saved100.pth"  # load last checkpoint
    dit.load_state_dict(torch.load(checkpoint_path))

    inference(dit,vae,val_dataloader, user_t=[1])

if __name__ == "__main__":
    main_inference()
    main()
