import os
import sys
import torch
import urllib3
import numpy as np
import torch.nn.functional as F

from typing import Dict
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm


from .config import device
from .training_functions import (
    CustomDataset,
    dataloader_from_folders
)

from .network import (
    UNet,
    Discriminator,
    weights_init,
    train,
    save_model,
    visualize_loss,
)

class TrainModel():
    def __init__(self,config:Dict):
        self.config=config

    def prepare_training(self):

        torch.manual_seed(self.config['seed'])

        if self.config['obtener_el_dataloader'] == True:
            ### CREAR EL DATALOADER
            self.dataloader = dataloader_from_folders(
                self.config['path_folder'],
                self.config['input_folder'],
                self.config['label_folder'],
                self.config['batch_size'],
                self.config['shuffle'],
                self.config['target_shape'],
                self.config['real_dim'],
                self.config['input_dim'],
            )
        else: pass

        if self.config['iniciar_modelo']:
            self.gen = UNet(self.config['input_dim'], self.config['real_dim']).to(device)
            self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.config['lr'])
            self.disc = Discriminator(self.config['input_dim'] + self.config['real_dim']).to(device)
            self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.config['lr'])

            if self.config['pretrained']:
                self.loaded_state = torch.load(self.config['model_weights'], map_location=torch.device(device))
                self.gen.load_state_dict(self.loaded_state["gen"])
                self.gen_opt.load_state_dict(self.loaded_state["gen_opt"])
                self.disc.load_state_dict(self.loaded_state["disc"])
                self.disc_opt.load_state_dict(self.loaded_state["disc_opt"])
                self.cur_step = int(self.config['model_weights'].split("_")[-1].split(".")[0])

            if not self.config['pretrained']:
                self.gen = self.gen.apply(weights_init)
                self.disc = self.disc.apply(weights_init)
                self.cur_step = 0
        else:
            self.cur_step = 0

    def train_model(self):
        gen, gen_opt, disc, disc_opt, cur_step, gen_loss, disc_loss = train(
            self.dataloader,
            self.config['n_epochs'],
            device,
            self.gen,
            self.disc,
            self.gen_opt,
            self.disc_opt,
            self.config['adv_criterion'],
            self.config['recon_criterion'],
            self.config['lambda_recon'],
            self.config['display_step'],
            self.config['RGB_fake'],
            self.config['RGB_real'],
            self.cur_step,
            self.config['input_dim'],
            self.config['real_dim'],
            self.config['target_shape'],
        )
        self.gen = gen
        self.gen_opt = gen_opt
        self.disc = disc
        self.disc_opt = disc_opt
        self.cur_step = cur_step
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss

    def save_trained_model(self):

        gen_loss_ = f"{self.config['gen_loss_step_']}{self.cur_step}.npy"
        disc_loss_ = f"{self.config['disc_loss_step_']}{self.cur_step}.npy"

        print(f'\nEL ULTIMO GRUPO DE ITERACION ES {self.cur_step-70000} - {self.cur_step}\n')

        save_model(
            self.gen,
            self.gen_opt,
            self.disc,
            self.disc_opt,
            self.cur_step,
            self.config['path_weights'],
        )

        path_gen_loss = f"{self.config['path_loss']}\{self.config['gen_loss_folder']}\{gen_loss_}"
        path_disc_loss = f"{self.config['path_loss']}\{self.config['disc_loss_folder']}\{disc_loss_}"
        
        np.save(path_gen_loss, self.gen_loss)
        np.save(path_disc_loss, self.disc_loss)

        print(f"\nSe almacenaron los pesos\n{path_gen_loss}\n{path_disc_loss}")

    def visualice_model_history(self):

        gen_p = f"{self.config['path_loss']}/{self.config['gen_loss_folder']}"
        disc_p = f"{self.config['path_loss']}/{self.config['disc_loss_folder']}"

        gen_losses = os.listdir(gen_p)
        disc_losses = os.listdir(disc_p)

        gen_losses = [np.load(f"{gen_p}/{path_}") for path_ in gen_losses if "npy" and "step" in path_]
        disc_losses = [np.load(f"{disc_p}/{path_}") for path_ in disc_losses if "npy" and "step" in path_]

        _gen_loss_ = np.concatenate(gen_losses, axis=0)
        _disc_loss_ = np.concatenate(disc_losses, axis=0)

        visualize_loss(
            _gen_loss_,
            _disc_loss_,
            step_bins = self.config['bins'],
            title="\nHistórico de perdidas\n"
        )

        visualize_loss(
            _gen_loss_,
            np.array([]),
            step_bins = self.config['bins'],
            title="\nHistórico de perdidas Generador\n"
        )

        visualize_loss(
            np.array([]),
            _disc_loss_,
            step_bins = self.config['bins'],
            title="\nHistórico de Discriminador\n"
        )
