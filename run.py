import requests
url = 'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/utils/datasets/celeba.py'
open('celeba.py', 'wb').write(requests.get(url).content)
url = 'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/data/celeba/list_attr_celeba.txt'
open('list_attr_celeba.txt', 'wb').write(requests.get(url).content)

from src.train import train
from src.datasets import CelebaCustomDataset_V2, CelebaCustomDatasetRef_V2
from torchvision import transforms
import torch
from src.models import Generator, MappingNetwork, StyleEncoder, Discriminator
from src.arguments import Arguments


# GET DATA
t_normalize = lambda x: x * 2 - 1
t_invnormalize = lambda x: (x + 1) / 2
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    t_normalize,
])

dataset = CelebaCustomDataset_V2(
    transform=transform,
    attr_file_path='list_attr_celeba.txt',
    crop=True
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, drop_last=True)

ref_dataset = CelebaCustomDatasetRef_V2(
    transform=transform,
    attr_file_path='list_attr_celeba.txt',
    crop=True
)
ref_loader = torch.utils.data.DataLoader(ref_dataset, batch_size=32, drop_last=True)


# GET MODELS
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
style_enc = StyleEncoder().to(device)
mapping = MappingNetwork().to(device)
gen = Generator().to(device)
disc = Discriminator().to(device)
nets = {
    'generator': gen,
    'discriminator': disc,
    'style': style_enc,
    'mapping': mapping
}

# GET OPTIMIZERS
beta = [0, 0.99]
lr = 1e-4
g_optim = torch.optim.Adam(gen.parameters(), lr=lr, betas=beta)
d_optim = torch.optim.Adam(disc.parameters(), lr=lr, betas=beta)
s_optim = torch.optim.Adam(style_enc.parameters(), lr=lr, betas=beta)
m_optim = torch.optim.Adam(mapping.parameters(), lr=1e-6, betas=beta)
optims = {
    'generator': g_optim,
    'discriminator': d_optim,
    'style': s_optim,
    'mapping': m_optim
}

args = Arguments

train(dataloader, ref_loader, nets, optims, args, num_epochs=10)
