import torch
import matplotlib.pyplot as plt
import math

LEARNING_RATE = 0.0001

INFER_SIZE = 4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('medium')  # or 'high'

LOG_EVERY = 1
CFG_SCALE = 0.0

def inference(dit, vae, val_dataloader, user_t = [1, 2, 4, 8, 16, 32, 128]):
    dit.eval()

    images, _ = next(iter(val_dataloader))
    images = images[:INFER_SIZE].to(DEVICE) # .to(DEVICE)
    with torch.no_grad():
        latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
    text_input=["The woman wearing glasses", " The man with black hair. The man wearing glasses.", "The woman is smiling", "The man wearing glasses"]
    # all start from same noise
    eps = torch.randn_like(latents).to(DEVICE)

    denoise_timesteps_list = user_t
    # run at varying timesteps
    for i, denoise_timesteps in enumerate(denoise_timesteps_list):
        all_x = []
        delta_t = 1.0 / denoise_timesteps  # i.e. step size

        x = eps

        for ti in range(denoise_timesteps):
            # t should in range [0,1]
            t = ti / denoise_timesteps

            t_vector = torch.full((eps.shape[0],), t).to(DEVICE)
            dt_base = torch.ones_like(t_vector).to(DEVICE) * math.log2(denoise_timesteps)

            with torch.no_grad():
                v = dit.forward(x, t_vector, dt_base, text_input)

            x = x + v * delta_t

            if denoise_timesteps <= 8 or ti % (denoise_timesteps // 8) == 0 or ti == denoise_timesteps - 1:
                with torch.no_grad():
                    decoded = vae.decode(x / vae.config.scaling_factor)[0]

                decoded = decoded.to("cpu")

                all_x.append(decoded)

        if (len(all_x) == 9):
            all_x = all_x[1:]

        # generated images
        all_x = torch.stack(all_x)

        def process_img(img):
            # normalize in range [0,1]
            img = img * 0.5 + 0.5
            img = torch.clip(img, 0, 1)
            img = img.permute(1, 2, 0)
            return img

        fig, axs = plt.subplots(8, 4, figsize=(30, 30))
        for t in range(min(8, all_x.shape[0])):
            for j in range(4):
                axs[t, j].imshow(process_img(all_x[t, j]), vmin=0, vmax=1)

        fig.savefig(f"gen_image.png")


        plt.close()