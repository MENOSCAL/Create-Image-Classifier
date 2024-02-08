import os
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

resnet50 = models.resnet50(pretrained = True)
densenet121 = model = models.densenet121(pretrained=True)

models_choose = {'resnet50': resnet50, 'densenet121': densenet121}
in_features_models = {'resnet50': 2048, 'densenet121': 1024}
adam_param = {'resnet50': 'fc', 'densenet121': 'classifier'}
folders = ['train', 'valid', 'test']

def data_dir_exists(data_dir):
    data_dir_bool = False
    if os.path.isdir(data_dir):
        for folder in folders:
            if not os.path.isdir(os.path.join(data_dir, folder)):
                print("The directory does not exist.")
                return data_dir_bool
        data_dir_bool = True
    else:
        print("The directory does not exist.")
    return data_dir_bool

def loading_data(data_dir):
    train_dir = os.path.join(data_dir, folders[0])
    valid_dir = os.path.join(data_dir, folders[1])
    test_dir = os.path.join(data_dir, folders[2])
    data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                    transforms.RandomResizedCrop(224),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])]),
                        'valid': transforms.Compose([transforms.Resize(255),
                                                     transforms.CenterCrop(224),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                                          [0.229, 0.224, 0.225])]),
                        'test': transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
    }

    image_datasets = {'train': datasets.ImageFolder(train_dir, transform = data_transforms['train']),
                      'valid': datasets.ImageFolder(valid_dir, transform = data_transforms['valid']),
                      'test': datasets.ImageFolder(test_dir, transform = data_transforms['test'])
    }

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size = 64, shuffle = True),
                   'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size = 64),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64)
    }
    return image_datasets, dataloaders

def validation(model, test_loader, criterion, device):
    accuracy = 0
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for steps, (inputs, labels) in enumerate(test_loader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels).to("cpu")
            test_loss += batch_loss.item()
            
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.float)).item()

            if steps == len(test_loader):
                break
    return accuracy, steps, test_loss

def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs, print_every_bool=False):
    print("Training a new network on the", device)
    every_steps = 34
    print_every = every_steps if print_every_bool else len(train_loader)
    train_losses, valid_losses,  accuracy = [], [], []
    
    for epoch in range(epochs):
        steps = 0
        running_loss = 0
        model.train()
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if steps % print_every == 0:
                model.eval()
                accuracy_valid, size_valid, valid_loss = validation(model, valid_loader, criterion, device)
                
                if len(train_losses) == epoch + 1:
                    train_losses[-1] = running_loss/print_every
                    valid_losses[-1] = valid_loss/len(valid_loader)
                    accuracy[-1] = accuracy_valid/size_valid
                else:
                    train_losses.append(running_loss/print_every)
                    valid_losses.append(valid_loss/len(valid_loader))
                    accuracy.append(accuracy_valid/size_valid)
                
                print("Epoch: {}/{}..".format(epoch + 1, epochs),
                      "Train loss: {:.3f}..".format(train_losses[-1]),
                      "Validate loss: {:.3f}..".format(valid_losses[-1]),
                      "Validate accuracy: {:.2f}%".format(accuracy[-1]*100))
                running_loss = 0
                model.train()
    return train_losses, valid_losses, accuracy

def main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, bool_gpu):
    dropout = 0.2
    out_features = 102
    if bool_gpu:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    data_dir_bool = data_dir_exists(data_dir)
    if not data_dir_bool:
        return
    image_datasets, dataloaders = loading_data(data_dir)

    model_name = arch
    model = models_choose[model_name]
    in_features = in_features_models[model_name]
    for param in model.parameters():
        param.requires_grad = False

    layers = nn.Sequential(nn.Linear(in_features, hidden_units),
                           nn.ReLU(),
                           nn.Dropout(dropout),
                           nn.Linear(hidden_units, out_features),
                           nn.LogSoftmax(dim=1))

    if model_name == 'resnet50':
        model.fc = layers
    elif model_name == 'densenet121':
        model.classifier = layers

    optimizer = optim.Adam(getattr(model, adam_param[model_name]).parameters(), lr = learning_rate)
    criterion = nn.NLLLoss().to(device)

    model.to(device)
    train_losses, valid_losses, accuracy = train(model, dataloaders['train'], dataloaders['valid'],
                                                 criterion, optimizer, device, epochs)
    
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {'input_size': in_features,
                  'output_size': out_features,
                  'hidden_layers': hidden_units,
                  'dropout': dropout,
                  'epochs': epochs,
                  'model_name': model_name,
                  'learning_rate': learning_rate,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict()
    }
    
    file_name = 'checkpoint_' + model_name + '.pth'
    if save_dir == None:
        torch.save(checkpoint, file_name)
    else:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(checkpoint, os.path.join(save_dir, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type = str, help = 'Path to DataSet')
    parser.add_argument('--save_dir', type = str, help = 'Path to Save Checkpoints')
    parser.add_argument('--arch', type = str, default = 'resnet50', choices = ['resnet50', 'densenet121'],
                        help = 'Choose Architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.003,
                        help = 'Set Hyperparameters Learning Rate')
    parser.add_argument('--hidden_units', type = int, default = 512,
                        help = 'Set Hyperparameters Hidden Units')
    parser.add_argument('--epochs', type = int, default = 6,
                        help = 'Set Hyperparameters Epochs')
    parser.add_argument('--gpu', action = 'store_true', help = 'GPU Enable')

    args = parser.parse_args()
    main(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)