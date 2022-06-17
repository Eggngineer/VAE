from cProfile import label
import sys
from tabnanny import check
sys.dont_write_bytecode = True

import torch
import yaml
import wandb
import numpy as np
from pathlib import Path
from model.vae import VAE
import matplotlib.pyplot as plt
from matplotlib import cm

from data . dataloader import MNIST_Loader

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def load_yml(yml_path):
    with open(yml_path) as tyaml:
        yml = yaml.safe_load(tyaml)
        return yml


def show_imgs(data):
    data = np.asarray(data * 255, dtype=np.uint8)
    plt.figure("title", figsize=(len(data), 1))
    for i in range(len(data)):
        plt.subplot(1, len(data), i+1)
        plt.axis('off')
        plt.imshow(data[i].transpose(1, 2, 0), cmap=cm.gray, interpolation='nearest')
    plt.pause(100)
    plt.close()

def evals(conf):
    image_size = conf['image_size']
    h_dim = conf['h_dim']
    z_dim = conf['z_dim']

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    BASE_DIR = Path('.')

    WEIGHT_DIR = BASE_DIR / conf['weight']
    model_path = WEIGHT_DIR / (conf['model_name']+'.pth')

    model = VAE(image_size,h_dim,z_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    with torch.no_grad():
        loader = MNIST_Loader(conf).__iter__()
        data, labels = loader.next()
        show_imgs(data[labels==3])
        x = data.to(device).view(-1, image_size).to(torch.float32)
        _,_,z = model.encoder(x)
        out = model.decoder(z)
        out = out[labels == 3]


        num1 = torch.mean(z[labels == 0],0)
        num2 = torch.mean(z[labels == 5],0)
        divn = len(z)-1
        diff = (num2 - num1) / divn
        orig_z = torch.zeros(z.shape)
        for i in range(divn+1):
            orig_z[:][i] = num1 + i * diff
        
        orig_out = model.decoder(orig_z)[:][::11]
        orig_out = orig_out.view(-1,1,28,28)
        orig_out = orig_out.cpu().detach().numpy()
        orig_out = np.array(orig_out)
        show_imgs(orig_out)
        






    out = out.view(-1,1,28,28)
    out = out.cpu().detach().numpy()
    out = np.array(out)

    return z, out

def main():
    config = load_yml('yml/test.yml')
    # wandb.init(project=config['project_name'], config=config, name=config['test_name'])
    # config = wandb.config

    torch.backends.cudnn.deterministic = True
    fix_seed(config['seed'])

    BASE_DIR = Path('.')
    WEIGHT_DIR = BASE_DIR
    TEST_YML = BASE_DIR

    WEIGHT_DIR = WEIGHT_DIR / config['weight']
    TEST_YML = TEST_YML / config['yml']

    z, out = evals(conf=config)
    # for i in range(len(out)):
    #     output_images = wandb.Image(out[i],caption=str(z[i]))
    #     wandb.log({'output':output_images})

    show_imgs(out)

if __name__ == '__main__':
    main()