import os
import argparse
import json
from PIL import Image

import numpy as np

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

resnet50 = models.resnet50(pretrained = True)
densenet121 = model = models.densenet121(pretrained=True)

models_choose = {'resnet50': resnet50, 'densenet121': densenet121}
adam_param = {'resnet50': 'fc', 'densenet121': 'classifier'}

def load_checkpoint(file_path, device_name):
    checkpoint = torch.load(file_path, map_location = device_name)
    
    model_name = checkpoint['model_name']
    model = models_choose[model_name]
    for param in model.parameters():
        param.requires_grad = False
    
    layers = nn.Sequential(nn.Linear(checkpoint['input_size'], checkpoint['hidden_layers']),
                           nn.ReLU(),
                           nn.Dropout(checkpoint['dropout']),
                           nn.Linear(checkpoint['hidden_layers'], checkpoint['output_size']),
                           nn.LogSoftmax(dim=1))
    
    if model_name == 'resnet50':
        model.fc = layers
    elif model_name == 'densenet121':
        model.classifier = layers
    
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = optim.Adam(getattr(model, adam_param[model_name]).parameters(), lr = checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']
    epochs = checkpoint['epochs']
    return model, optimizer, epochs

def process_image(image):
    width, height = image.size
    if width < height:
        new_width = 256
        new_height = int((height / width)*new_width)
    else:
        new_height = 256
        new_width = int((width / height)*new_height)
    img = image.resize((new_width, new_height))

    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def label_folder(model, cls):
    class_to_idx = model.class_to_idx
    idx_to_class = {label: key for key, label in class_to_idx.items()}
    return idx_to_class[cls]

def predict(image_path, model, device, topk):
    print("Predict flower name from an image on the", device)
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path)
        input = process_image(image)
        
        input = torch.from_numpy(input).float()
        input = input.unsqueeze(0)
        input = input.to(device)
        
        logps = model.forward(input)
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)
        top_p, top_class = top_p.to("cpu"), top_class.to("cpu")
        top_p, top_class = top_p.numpy().flatten(), top_class.numpy().flatten()
        labels = [label_folder(model, cls) for cls in top_class]
    return top_p, labels

def main(image_path, checkpoint, top_k, category_names, bool_gpu):
    if bool_gpu:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device_name = "cpu"
    device = torch.device(device_name)
    
    if not os.path.isfile(category_names):
        category_names = 'cat_to_name.json'

    with open(category_names, 'r') as f:
        folder_to_name = json.load(f)
    
    model_load, optimizer_load, epochs_load = load_checkpoint(checkpoint, device_name)
    
    model_load.to(device)
    probs, classes = predict(image_path, model_load, device, top_k)
    class_to_name = [folder_to_name[cls] for cls in classes]

    predict_flowers = list(zip(class_to_name, probs))
    for predict_flower in predict_flowers:
        print(predict_flower)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type = str, help = 'Path to Image')
    parser.add_argument('checkpoint', type = str,
                        help = 'Path to Loading Checkpoints')
    parser.add_argument('--top_k', type = int, default = 1,
                        help = 'K Most Likely Classes')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'Mapping of categories to real names')
    parser.add_argument('--gpu', action = 'store_true', help = 'GPU Enable')
    
    args = parser.parse_args()
    main(args.input, args.checkpoint, args.top_k, args.category_names, args.gpu)