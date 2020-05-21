import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision import transforms, utils
from PIL import Image

from dataset import read_picture
from style_AE import *




if __name__ == "__main__":


    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    input_img = read_picture('style_samples/content_pictures/sample.jpg', resize=False).cuda()

    ae_model = Style_AutoEncoder().to(device)

    print('Weights Loading....')
    ae_model.load_state_dict(torch.load('models/AutoEncoder/ae_style_1.pth'))

    print("Success!")

    out = ae_model(input_img)
    


        
    