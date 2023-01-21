import torch
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_size = 128
batch_size = 64
latent_dim = 200

MEAN = torch.tensor([0.5, 0.5, 0.5])
STD = torch.tensor([0.5, 0.5, 0.5])

transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(MEAN, STD)])

denormalization = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())

SEX_INDEX = 21 - 1
EYEGLASSES_INDEX = 16 - 1
