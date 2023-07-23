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

parser = argparse.ArgumentParser(
    prog="Train.py",
    description="List the arguments of a program",
    epilog="Thanks for using %(prog)s! :)",
)

parser.add_argument('dir',type=str)
parser.add_argument('--save_dir',type=str)
parser.add_argument('--arch',type=str)
parser.add_argument('--learning_rate',type=float)
parser.add_argument('--hidden_units',type=int)
parser.add_argument('--epochs',type=int)
parser.add_argument('--gpu',action='store_true')


args=parser.parse_args()

if os.path.isdir(args.dir):
    data_dir = args.dir
    print(f'Directory Found {data_dir}')
else:
    print("Provide valid Directory Path")
    sys.exit("Program is shutting Down!!")

if args.save_dir is None:
    print(f"Check Point will be saved in {os.getcwd()}")
    ckpt_path=os.getcwd()
else:
    if os.path.isdir(args.save_dir):
        print(f"Check Point will be saved in {args.save_dir}")
        ckpt_path=args.save_dir
    else:
        print("Provide valid path to store Checkpoint Path")
        sys.exit("Program is shutting Down!!")
        
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transform=transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

val_test_trans=transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
test_transform = transforms.Compose([transforms.Resize(256), 
                                         transforms.CenterCrop(224), 
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 


image_datasets =[datasets.ImageFolder(train_dir, transform=train_transform),
datasets.ImageFolder(valid_dir, transform=val_test_trans),
datasets.ImageFolder(test_dir, transform=test_transform)]


dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[1], batch_size=64, shuffle=True),
               torch.utils.data.DataLoader(image_datasets[2], batch_size=64, shuffle=True)]


if args.arch is None:
    model_name='vgg16'
else:
    model_name=args.arch
model=models.vgg16(pretrained = True)
print(f"using {model_name}")

if args.hidden_units is None:
    hidden_layer=256
else:
    hidden_layer=args.hidden_units 
    

for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(
          nn.Linear(25088, 256),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(256, 102),
          nn.LogSoftmax(dim = 1)
        )

model.classifier = classifier

total_params = sum(p.numel() for p in model.parameters())
print(f'Total_no_of_parameters: {total_params}')
trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total_no_of_trainable_parameters: {trainable_total_params}')

device="cpu"
if args.gpu:
    if torch.cuda.is_available():
        print("Model is running on GPU")
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print("No GPU Found,Hence")
        print("Model is running on CPU!")
    
criterion = nn.NLLLoss()
if args.learning_rate  is None:
    lr=0.001
else:
    lr=args.learning_rate 
    
optimizer = optim.Adam(model.classifier.parameters(), lr =lr)
model.to(device)

def train_model(model,criterion,optimizer,num_epochs=10):
    start=time.time()
    steps=0
    print_every=30
    train_loss = 0.0
    train_losses, val_losses = [], []
    train_acc,val_acc=[],[]
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        for inputs, labels in dataloaders[0]:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if steps % print_every == 0:
                val_loss = 0
                val_accuracy = 0
                
                #setting in eval mode
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in dataloaders[1]:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()

                        # Calculate validation accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print( f"Training loss: {train_loss/print_every:.3f},"
                  f"Validation loss: {val_loss/len(dataloaders[1]):.3f}, "
                  f"Validation accuracy: {val_accuracy/len(dataloaders[1]):.3f} \n")
                train_loss = 0.0
                model.train()
    time_elapsed = time.time() - start
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    
    return model

if args.epochs  is None:
    epochs=10
else:
    epochs=args.epochs  

model=train_model(model,criterion,optimizer,num_epochs=epochs)

checkpoint = {
    'epochs': epochs,
    'learning_rate': lr,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict(),
    'class_to_idx': image_datasets[0].class_to_idx
}

#torch.save(checkpoint, 'checkpoint_model.pth')

torch.save(checkpoint, os.path.join(ckpt_path,'checkpoint_model.pth'))

print('CheckPoint Saved')

print("Training Finished \n closing program")
