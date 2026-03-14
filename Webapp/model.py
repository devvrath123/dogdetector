import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import csv

def load_model(path, num_classes):
    model = models.convnext_small(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model

def create_transforms():
    validation_transforms = transforms.Compose([
        transforms.Resize(236, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    return validation_transforms

def predict_top5(path, model, class_names, validation_transforms):
    img = Image.open(path).convert('RGB')
    img_tensor = validation_transforms(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)[0]
        top5, indices = torch.topk(probs, 5)
        print('Top 5 predictions:')
        for i in range(5):
            idx = indices[i].item()
            p = top5[i].item() * 100
            breed = class_names[idx]
            print(f"{i+1}. {breed} - {p:.2f}%")

def predict(path, model, class_names, validation_transforms):
    img = Image.open(path).convert('RGB')
    img_tensor = validation_transforms(img).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        prob, pred_idx = torch.max(probs, 1)
    breed = (class_names[pred_idx.item()])
    return breed, prob.item() * 100

def get_breeds(path, skip=True):
    with open(path, 'r') as f:
        file = csv.reader(f)
        if skip:
            next(file)
        breeds = []
        for line in file:
            breeds.append(''.join(line[0]))
    return breeds

def get_breeds_dict(path):
    with open(path, 'r') as f:
        file = csv.reader(f)
        next(file)
        breed_dict = {}
        for line in file:
            breed_dict[line[0]] = int(line[1])
    return breed_dict

def update_breeds(path, breeds_dict):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['breed', 'count'])
        for key, value in breeds_dict.items():
            writer.writerow([key, value])

def reset_breeds_count(target, init_path):
    init_breeds = get_breeds(init_path, skip=False)
    with open(target, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['breed', 'count'])
        for breed in init_breeds:
            writer.writerow([breed, 0])