# Model Defenition


# ------------------ Importing Libraries ------------------ #
import os
import wget


# ------------------ Model Class Defenition ------------------ #
class Model():
    """
    Model: 
        Class used to retrieve latest models and store relavent parameters.

    Attributes:
        name (str): Name of the model (either 'lightning' or 'thunder').
        url (str): URL of the TFLite model file for the given model.
        input_dim (Tuple[int, int]): Dimensions of the input image for the given model.
        file_path (str): Local file path to the downloaded TFLite model file.
    """
    
    def __init__(self, name):
        """
        __init__:
            Initializes a Model instance with the given name.

        Args:
            name (str): Name of the model (either 'lightning' or 'thunder').
        """
        if name == 'lightning':
            self.url = 'https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite'
            self.input_dim = (192, 192)
        elif name == 'thunder':
            self.url = 'https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite'
            self.input_dim = (256, 256)
        else:
            raise ValueError("Invalid model name. Valid options are 'lightning' or 'thunder'.")
        
        self.name = name
        self.file_path = ''


    def download_model(self):
        """
        download_model:
            Downloads the TFLite model file for the given model if it hasn't been downloaded yet.
        """
        file_dir = os.path.join( os.getcwd(), f'{self.name}.tflite' )

        if os.path.exists(file_dir):
            print("File Already Downloaded")
        else:
            print("Downloading File")
            wget.download(self.url, file_dir)
        
        self.file_path = file_dir