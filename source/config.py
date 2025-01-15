import os
import torch
import numpy as np
from torch import nn
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"El dispositivo es {device}")

class Configuration():
    def __init__(self):
        self.n_epochs: int = 1
        self.step: int = 0
        self.path_folder: str = os.getenv("PATH_FOLDER")
        self.path_weights: str = os.getenv("PATH_WEIGHTS")
        self.weight_name: str = "pix2pix_step_"
        self.obtener_el_dataloader : bool = True
        self.iniciar_modelo : bool = True
        self.pretrained : bool = True
        self.seed: int = np.random.randint(0,10000)
        self.RGB_fake: tuple = (0,0,0)
        self.RGB_real: tuple = (2,1,0)
        self.input_folder: str = "optico"
        self.label_folder: str = "radar"
        self.real_dim: str = 6
        self.adv_criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.recon_criterion: nn.L1Loss = nn.L1Loss()
        self.lambda_recon: int = 200
        self.input_dim: int = 2
        self.display_step: int = 10000
        self.batch_size: int = 1
        self.shuffle: bool = True
        self.lr: float = 0.0002 
        self.target_shape: int = 256
        self.path_loss: str = os.getenv("PATH_LOSS")
        self.gen_loss_step_ : str = "gen_loss_step_"
        self.disc_loss_step_: str = "disc_loss_step_"
        self.gen_loss_folder:str = "gen_loss"
        self.disc_loss_folder:str = "disc_loss"
        self.bins: int = 10000

    def to_dict(self):
        return {
            "n_epochs":self.n_epochs,
            "step" : self.step,
            "path_folder":self.path_folder,
            "path_weights":self.path_weights,
            "model_weights":f"{self.path_weights}{self.weight_name}{self.step}.pth",
            "obtener_el_dataloader":self.obtener_el_dataloader,
            "iniciar_modelo":self.iniciar_modelo,
            "pretrained":self.pretrained,
            "seed":self.seed,
            "RGB_fake":self.RGB_fake,
            "RGB_real":self.RGB_real,
            "input_folder":self.input_folder,
            "label_folder":self.label_folder,
            "real_dim":self.real_dim,
            "adv_criterion":self.adv_criterion,
            "recon_criterion":self.recon_criterion,
            "lambda_recon":self.lambda_recon,
            "input_dim":self.input_dim,
            "display_step":self.display_step,
            "batch_size":self.batch_size,
            "shuffle":self.shuffle,
            "lr":self.lr,
            "target_shape":self.target_shape,
            "path_loss": self.path_loss,
            "gen_loss_step_":self.gen_loss_step_,
            "disc_loss_step_":self.disc_loss_step_,
            "gen_loss_folder":self.gen_loss_folder,
            "disc_loss_folder":self.disc_loss_folder,
            "bins":self.bins
        }


