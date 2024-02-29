#import Train
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy


def modele_vgg(model,img,device) :
    """model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 2)
    poids = torch.load('/root/catkin_ws/src/imredd_pkg/src/modele_poids.pt')
    model.load_state_dict(poids)
    device = torch.device("cuda:0")
    model.to(device)"""


    image_vecteur = numpy.frombuffer(img.data, dtype="uint8")
    image = image_vecteur.reshape(240,424,3)
    image = Image.fromarray(image)

# Appliquer les transformations à l'image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    return model(image)[0].detach().cpu().numpy()

