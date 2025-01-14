
import os
import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple
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
        """
        Inicializa el conjunto de datos personalizado para el entrenamiento del modelo.

        Esta función construye el conjunto de datos cargando rutas de archivos de imágenes de entrada y etiquetas desde las carpetas especificadas.
        Se asegura de que los archivos sean de tipo .tif y se ordenan alfabéticamente.

        Args:
            input_folder (str): Ruta de la carpeta que contiene las imágenes de entrada.
            label_folder (str): Ruta de la carpeta que contiene las imágenes de etiqueta correspondientes a las imágenes de entrada.
            target_shape (int, optional): Tamaño objetivo de las imágenes (altura y anchura). Por defecto es 256.
            input_channels (int, optional): Número de canales de las imágenes de entrada. Por defecto es 6.
            label_channels (int, optional): Número de canales de las imágenes de etiqueta. Por defecto es 2.

        Raises:
            TypeError: Si los argumentos no son del tipo esperado.
            FileNotFoundError: Si las carpetas o archivos no se encuentran.

        """
        self.input_files = sorted([f"{input_folder}\{f}" for f in os.listdir(input_folder) if f.endswith('.tif')])
        self.label_files = sorted([f"{label_folder}\{f}" for f in os.listdir(label_folder) if f.endswith('.tif')])

        self.target_shape = target_shape
        self.input_channels = input_channels
        self.label_channels = label_channels

    def __len__(self) -> int:
        """
        Devuelve el número total de pares de imágenes de entrada y etiqueta disponibles en el conjunto de datos.

        Esta función retorna la longitud del atributo 'self.input_files', que contiene la lista de rutas a las imágenes de entrada.
        Dado que se asume que cada imagen de entrada tiene una imagen de etiqueta correspondiente, la longitud de 'self.input_files' representa el número total de ejemplos de entrenamiento disponibles.

        Returns:
            int: El número total de ejemplos de entrenamiento.
        """
        return len(self.input_files)

    def __getitem__(
            self, 
            idx: int
        ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Devuelve un par de tensores de imagen de entrada y etiqueta para un índice dado.

        Esta función carga las imágenes de entrada y etiqueta correspondientes al índice especificado del conjunto de datos.
        Utiliza la librería rasterio para abrir los archivos .tif, leer sus datos y convertirlos a tensores de PyTorch.
        Normaliza los valores de los píxeles de las imágenes dividiéndolos por 255.0 para obtener valores entre 0 y 1.
        Maneja las excepciones (como archivos no encontrados o problemas de lectura) estableciendo tensores de ceros si falla la lectura de una imagen.
        Los tensores de salida representan las imágenes de entrada y etiqueta correspondientes al índice dado, normalizadas y convertidas a tipo flotante.

        Args:
            idx (int): El índice del par de imagen de entrada y etiqueta a obtener.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Un par de tensores de PyTorch que representan la imagen de entrada y la imagen de etiqueta, respectivamente.  Devuelve tensores de ceros si hay un error en la lectura de los archivos.
        
        Raises:
            IndexError: Si el índice `idx` está fuera del rango de los archivos disponibles.
            rasterio.RasterioIOError: Si hay un error al abrir o leer un archivo de imagen con rasterio.
            TypeError: Si los datos leídos no se pueden convertir a un tensor de PyTorch.
        """
        
        # Obtener las rutas de las imágenes de entrada y etiqueta correspondientes al índice.
        input_path = self.input_files[idx]
        label_path = self.label_files[idx]

        try:
            # Abrir la imagen de entrada con rasterio.
            with rasterio.open(input_path) as input_src:
                # Leer los datos de la imagen de entrada.
                input_img = input_src.read()

            # Convertir los datos a un tensor de PyTorch de tipo flotante.
            input_img = torch.FloatTensor(input_img)
            # Normalizar los valores de los píxeles dividiendo por 255.0.
            input_img = input_img / 255.0
        except:
            input_img = torch.zeros(
                self.input_channels,
                self.target_shape,
                self.target_shape
            )

        try:
            # Abrir la imagen de etiqueta con rasterio.
            with rasterio.open(label_path) as label_src:
                # Leer los datos de la imagen de etiqueta.
                label_img = label_src.read()

            # Convertir los datos a un tensor de PyTorch de tipo flotante.
            label_img = torch.FloatTensor(label_img)
            # Normalizar los valores de los píxeles dividiendo por 255.0.
            label_img = label_img / 255.0
        except:
            label_img = torch.zeros(
                self.label_channels,
                self.target_shape,
                self.target_shape
            )
        # Retornar el par de tensores de imagen de entrada y etiqueta.
        return input_img, label_img

def dataloader_from_folders(
        path_folder: str,
        input_folder: str,
        label_folder: str,
        batch_size: int,
        shuffle: bool,
        target_shape: int = 256,
        input_channels: int = 6,
        label_channels: int = 2,
    ) -> DataLoader:
    """
    Crea un DataLoader para cargar datos de imágenes de entrada y etiqueta desde carpetas especificadas.

    Esta función construye un conjunto de datos personalizado y lo convierte en un DataLoader para cargar imágenes de entrada y etiqueta.
    Es crucial para el entrenamiento de modelos de aprendizaje automático, especialmente en tareas como la generación de imágenes o la segmentación.

    Args:
        path_folder: Ruta de la carpeta principal que contiene las subcarpetas de entrada y etiqueta.
        input_folder: Nombre de la subcarpeta que contiene las imágenes de entrada.
        label_folder: Nombre de la subcarpeta que contiene las imágenes de etiqueta correspondientes a las de entrada.
        batch_size: Tamaño del lote de imágenes que se cargarán en cada iteración del DataLoader.
        shuffle: Booleano que indica si las imágenes deben desordenarse aleatoriamente antes de cargarlas.
        target_shape: Tamaño deseado de las imágenes (altura y anchura). Por defecto es 256.
        input_channels: Número de canales de la imagen de entrada. Por defecto es 6.
        label_channels: Número de canales de la imagen de etiqueta. Por defecto es 2.

    Returns:
        DataLoader: Un objeto DataLoader que carga los datos de imagen.

    Raises:
        TypeError: Si los argumentos no son del tipo esperado.
        FileNotFoundError: Si las carpetas o archivos especificados no existen.
        ValueError: Si el tamaño del lote es inválido o si hay un problema con los canales de las imágenes.
        Exception: Maneja errores generales durante la lectura de los archivos.

    Descripción detallada:

    1. Construye las rutas completas para las carpetas de entrada y etiqueta.
    2. Crea un objeto CustomDataset, pasando las rutas, el tamaño deseado de las imágenes, y el número de canales de entrada y etiqueta.
    3. Crea un objeto DataLoader a partir del CustomDataset, especificando el tamaño del lote y si se deben mezclar los datos.
    4. Imprime un mensaje indicando el número de imágenes en el conjunto de datos.
    5. Obtiene un lote de datos (una muestra) del DataLoader para verificar la forma de los tensores.
    6. Imprime las formas de los tensores de entrada y etiqueta del lote de muestra.
    7. Retorna el objeto DataLoader.

    Ejemplo de uso:
    dataloader = dataloader_from_folders("ruta/a/la/carpeta", "optico", "radar", 1, True)
    """
    # Construye las rutas completas para las carpetas de entrada y etiqueta.
    input_ = f"{path_folder}\{input_folder}"
    label_ = f"{path_folder}\{label_folder}"
    # Crea un objeto CustomDataset, pasando las rutas, el tamaño deseado de las imágenes, y el número de canales de entrada y etiqueta.
    custom_dataset = CustomDataset(
        input_, 
        label_,
        target_shape,
        input_channels,
        label_channels
    )
    # Crea un objeto DataLoader a partir del CustomDataset, especificando el tamaño del lote y si se deben mezclar los datos.
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
