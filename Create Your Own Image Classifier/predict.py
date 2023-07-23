import numpy as np
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image
import argparse
import os
import sys
import json


val_test_trans=transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


parser = argparse.ArgumentParser(
    prog="Predict.py",
    description="List the arguments of a program",
    epilog="Thanks for using %(prog)s! :)",
)

parser.add_argument('img_path',type=str)
parser.add_argument('checkpoint',type=str)
parser.add_argument('--category_names',type=str)
parser.add_argument('--top_k',type=int)
parser.add_argument('--gpu',action='store_true')

args=parser.parse_args()

if os.path.isfile(args.img_path):
    image_path = args.img_path
    print(f'image Found {image_path}')
else:
    print("Provide valid Image Path")
    sys.exit("Program is shutting Down!!")

    
if os.path.isfile(args.checkpoint):
    ckpt_path = args.checkpoint
    print(f'Checkpoint Found {ckpt_path}')
else:
    print("Provide valid CheckPoint Path")
    sys.exit("Program is shutting Down!!")
    
    
show_cat_names=True
if args.category_names is None:
    show_cat_names=False
else:
    if os.path.isfile(args.category_names):
        cat_path = args.category_names
        print(f'category_names Found {cat_path}')
    else:
        print("Provide valid category_names Path")
        sys.exit("Program is shutting Down!!")
    
    
device="cpu"
if args.gpu:
    if torch.cuda.is_available():
        print("Model is running on GPU")
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("No GPU Found,Hence")
        print("Model is running on CPU!")


def load_checkpoint(filepath):
    classifier = nn.Sequential(
        nn.Linear(25088, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 102),
        nn.LogSoftmax(dim = 1))
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained = True)
    model.classifier = classifier
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

   
        
def process_image(image):
    image = Image.open(image)
    return val_test_trans(image)

def predict(image_path, model, topk,device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    with torch.no_grad():
        logps = model.forward(process_image(image_path).unsqueeze(0).to(device))
        logps=logps.cpu()
        ps = torch.exp(logps)
        probs, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return probs.numpy()[0], classes
        
        
model = load_checkpoint(ckpt_path)        
if args.top_k  is None:
    topk=5
else:
    topk=args.top_k          
predict_probs, predict_classes= predict(image_path, model,topk=topk,device=device)
print(predict_probs)
print(predict_classes)
            
if show_cat_names:

    with open(cat_path, 'r') as f:
        cat_to_name = json.load(f)
    classes = []
    for predict_class in predict_classes:
        classes.append(cat_to_name[predict_class])

    print(classes)
        
    

    

    

