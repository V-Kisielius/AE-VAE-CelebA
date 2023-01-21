import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import device, denormalization, latent_dim

def show_model(model, testloader, n=5):
    model.eval()
    with torch.no_grad():
        # take n * n random images from a test set
        test_images = next(iter(testloader))[0][:n*n]
        test_images = test_images.to(device)

        encoded = model.encode(test_images)
        decoded = model.decode(encoded)

        test_images = denormalization(test_images)
        decoded = denormalization(decoded)

        _, axs = plt.subplots(n, 2 * n, figsize=(25, 10))
        for i in range(n):
            for j in range(n):
                axs[i, j].imshow(test_images[i*n + j].permute(1, 2, 0).cpu().numpy())
                axs[i, j].axis('off')
                axs[i, n + j].imshow(decoded[i*n + j].permute(1, 2, 0).cpu().numpy())
                axs[i, n + j].axis('off')

def get_avg_by_labels(testset, label_idx_list):
    mask = torch.zeros((testset.attr.shape[1]), dtype=torch.bool)
    for idx in label_idx_list:
        mask[idx] = True
    mask = (testset.attr[:, mask] == torch.ones(len(label_idx_list), dtype=torch.int64)).prod(dim=1).squeeze()
    idxs = torch.nonzero(mask)
    result = torch.zeros(3, 128, 128)
    for i in tqdm(idxs, desc=f'Getting average image by labels {label_idx_list}'):
        i.squeeze_()
        result += testset[i][0]
    result /= len(idxs)
    return result

def plot_neighborhood(model, img, n=5, alpha=0.1):
    lat = model.encode(img)
    lat = lat.squeeze(0)
    lat = lat.unsqueeze(0)
    lat = lat.repeat(n, 1)
    lat = lat.unsqueeze(0)
    lat = lat.repeat(n, 1, 1)
    lat = lat.to(device)
    lat = lat + torch.randn_like(lat) * alpha

    rec = model.decode(lat.view(-1, latent_dim)).view(n, n, 3, 128, 128)
    _, axs = plt.subplots(n, n, figsize=(15, 15))
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(denormalization(rec[i, j]).permute(1, 2, 0).cpu().numpy())
            axs[i, j].axis('off')