import einops
import matplotlib.pyplot as plt

def display_img(img, h, w):
    reshaped = einops.rearrange(img, 'b c h w -> b h w c').cpu().numpy()
    fig, ax = plt.subplots(h, w, figsize=(10, 10))
    for i in range(h):
        for j in range(w):
            to_show = reshaped[h * i + j]
            ax[i, j].imshow(to_show)
            ax[i, j].axis('off')