
import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt

fromo typing import Tuple
from tqdm.notebook import tqdm
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(
            self, 
            input_folder:str, 
            label_folder:str,
            target_shape:int=256,
            input_channels:int=6,
            label_channels:int=2
        ):
        
        self.input_files = sorted([f"{input_folder}\{f}" for f in os.listdir(input_folder) if f.endswith('.tif')])
        self.label_files = sorted([f"{label_folder}\{f}" for f in os.listdir(label_folder) if f.endswith('.tif')])

        self.target_shape = target_shape
        self.input_channels = input_channels
        self.label_channels = label_channels

    def __len__(self)->int:
        return len(self.input_files)

    def __getitem__(
            self, 
            idx:int
        )->Tuple[torch.FloatTensor,torch.FloatTensor]:
        input_path = self.input_files[idx]
        label_path = self.label_files[idx]

        try:
            with rasterio.open(input_path) as input_src:
                input_img = input_src.read()

            input_img = torch.FloatTensor(input_img)
            input_img = input_img / 255.0
        except:
            input_img = torch.zeros(
                self.input_channels,
                self.target_shape,
                self.target_shape
            )

        try:
            with rasterio.open(label_path) as label_src:
                label_img = label_src.read()

            label_img = torch.FloatTensor(label_img)
            label_img = label_img / 255.0
        except:
            label_img = torch.zeros(
                self.label_channels,
                self.target_shape,
                self.target_shape
            )

        return input_img, label_img

def dataloader_from_folders(
        path_folder:str,
        input_folder:str,
        label_folder:str,
        batch_size:int,
        shuffle:bool,
        target_shape:int=256,
        input_channels:int=6,
        label_channels:int=2,
    )->DataLoader:

    ### Definir las rutas de las carpetas
    input_ = f"{path_folder}\{input_folder}"
    label_ = f"{path_folder}\{label_folder}"
    ### Obtener el dataloader
    custom_dataset = CustomDataset(
        input_, 
        label_,
        target_shape,
        input_channels,
        label_channels
    )

    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)
    print(f"El dataloader tiene {custom_dataset.__len__()} imagenes\n")

    input_data, label_data = next(iter(dataloader))
    print(input_data.shape)
    second_input = input_data[0]  # Segundo tensor de entrada
    second_label = label_data[0]  # Segundo tensor de etiqueta
    
    # Imprimir el tamaño de los tensores
    print("Shape del tensor de entrada:", second_input.shape)
    print("Shape del tensor de etiqueta:", second_label.shape)

    return dataloader
