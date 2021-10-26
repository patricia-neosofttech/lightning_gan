from collections import OrderedDict
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule
from pytorch_lightning.trainer import Trainer
from ds_class import HatchBackDataset
from img_to_npz import img_npz
from visualize_data import imgs, plot_images
from model import Generator, Discriminator
from argparse import ArgumentParser, Namespace


class BEGAN(LightningModule):

    def __init__(self,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 b1: float = 0.5,
                 b2: float = 0.999,
                 batch_size: int = 64, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size

        self.M = 1e+9

        # networks
        img_shape = (3,32,32)
        self.generator = Generator(latent_dim=self.latent_dim, img_shape=img_shape)
        self.discriminator = Discriminator(img_shape=img_shape)

        self.validation_z = torch.randn(8, self.latent_dim)

        self.example_input_array = torch.zeros(2, self.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return torch.mean(torch.abs(y_hat - y))

    def training_step(self, batch, batch_idx, optimizer_idx):
        # BEGAN hyper parameters
        gamma = 0.75
        lambda_k = 0.001
        k = 0.0

        imgs = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.latent_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images', grid, 0)

            # adversarial loss is binary cross-entropy
            fake_imgs = self(z)
            g_loss = self.adversarial_loss(self.discriminator(fake_imgs), fake_imgs)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            fake_imgs = self(z)

            real_loss = self.adversarial_loss(self.discriminator(imgs), imgs)
            fake_loss = self.adversarial_loss(self.discriminator(fake_imgs.detach()), fake_imgs.detach())

            # discriminator loss is the average of these
            d_loss = real_loss - k * fake_loss

            # ----------------
            # Update weights
            # ----------------
            diff = torch.mean(gamma * real_loss - fake_loss)
            # Update weight term for fake samples
            k = k + lambda_k * diff.item()
            k = min(max(k, 0), 1)  # Constraint to interval [0, 1]
            # Update convergence metric
            self.M = (d_loss + torch.abs(diff)).item()

            tqdm_dict = {'d_loss': d_loss, 'M': self.M}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # Add sync_dist=True to sync logging across all GPU workers
        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        # Add sync_dist=True to sync logging across all GPU workers
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        transpose_imgs = np.transpose(np.float32(imgs['arr_0']), (0, 3, 1, 2))
        self.dset = HatchBackDataset(transpose_imgs)  # passing the npz variable to the constructor class
        return DataLoader(self.dset, batch_size=self.batch_size)

    def on_epoch_end(self):
        z = self.validation_z.to(self.device)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)


def main(args: Namespace) -> None:
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = BEGAN(**vars(args))

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    
    trainer = Trainer(gpus=args.gpus,accelerator="ddp", max_epochs=100)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--gpus", type=int, default=-1, help="number of GPUs")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")


    hparams = parser.parse_args()

    main(hparams)
