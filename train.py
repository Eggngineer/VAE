import torch
from . data . dataloader import MNIST_Loader
from . model.vae import VAE
import torch.nn.functional as F

def loss_function(label, predict, mu, log_var):
  reconstruction_loss = F.binary_cross_entropy(predict, label, reduction='sum')
  kl_loss = -0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())
  vae_loss = reconstruction_loss + kl_loss
  return vae_loss, reconstruction_loss, kl_loss

def train(conf):
  epochs = conf['epochs']
  image_size = conf['image_size']
  h_dim = conf['h_dim']
  z_dim = conf['z_dim']
  lr = conf['learning_rate']

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  model = VAE(image_size, h_dim, z_dim).to(device)
  optim = torch.optim.Adam(model.parameters(), lr = lr)

  losses = []
  model.train()
  loader = MNIST_Loader(conf)

  for epoch in range(epochs):
    
    train_loss = 0

    for i, (x, labels) in enumerate(loader):

      x = x.to(device).view(-1, image_size).to(torch.float32)
      x_recon, mu, log_var, z = model(x)
      loss, recon_loss, kl_loss = loss_function(x, x_recon, mu, log_var)

      optim.zero_grad()
      loss.backward()
      optim.step()

      if (i+1) % 10 == 0:
        print(f'Epoch: {epoch+1}, loss: {loss: 0.4f}, reconstruction_loss: {recon_loss: 0.4f}, KL loss: {kl_loss: 0.4f}')
    
      losses.append(loss)    
    
  return losses, model


def main():
  conf = {
      'roots' : './data'
      , 'train' : True
      , 'downloads' : True
      , 'batch_size' : 128 
      , 'shuffle' : True
      , 'workers' : 0
      , 'epochs' : 10
      , 'image_size' : 784 #28*28
      , 'h_dim' : 32
      , 'z_dim' : 16
      , 'learning_rate' : 1e-3
  }

  loss, model = train(conf)

if __name__ == '__main__':
    main()