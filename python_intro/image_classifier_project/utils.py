import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms():
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    }
    return data_transforms

def load_data(data_dir, batch_size=64):
    train_dir = f"{data_dir}/train"
    valid_dir = f"{data_dir}/valid"
    test_dir = f"{data_dir}/test"
    
    transforms = get_transforms()
    
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=transforms["test"])
    }
    
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=batch_size, shuffle=False),
        "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False),
    }
    
    return image_datasets, dataloaders

def build_model(arch="vgg16", hidden_units=1024, output_classes=102):
    if arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        input_features = model.classifier[0].in_features
    elif arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
        input_features = model.classifier[0].in_features
    else:
        raise ValueError("Supported arch: vgg16, vgg13")
        
    for p in model.parameters():
        p.requres_grad = False
        
    model.classifier = nn.Sequential(
        nn.Linear(input_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(hidden_units, output_classes),
        nn.LogSoftmax(dim=1),
    )
    
    return model

def save_checkpoint(filepath, model, optimizer, epochs, arch, learning_rate, hidden_units, class_to_idx):
    checkpoint = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "classifier": model.classifier,
        "class_to_idx": class_to_idx,
        "optimizer_state_dict": optimizer.state_dict(),
        "epochs": epochs,
        "lr": learning_rate,
        "hidden_units": hidden_units,
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    arch = checkpoint["arch"]
    
    if arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif arch == "vgg13":
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
    else:
        raise ValueError("Unsupported architecture in checkpoint")

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.to(device)
    model.eval()
    return model

def process_image(image):
    width, height = image.size
    if width < height:
        new_w = 256
        new_h = int(256 * height / width)
    else:
        new_h = 256
        new_w = int(256 * width / height)

    image = image.resize((new_w, new_h))

    left = (new_w - 224) / 2
    top = (new_h - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))

    np_image = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return np_image

def predict(image_path, model, topk=5, device="cpu"):
    image = Image.open(image_path).convert("RGB")
    np_img = process_image(image)
    img_tensor = torch.from_numpy(np_img).unsqueeze(0).float().to(device)

    model.eval()
    with torch.no_grad():
        logps = model(img_tensor)
        ps = torch.exp(logps)
        top_p, top_idx = ps.topk(topk, dim=1)

    top_p = top_p.squeeze(0).cpu().numpy()
    top_idx = top_idx.squeeze(0).cpu().numpy()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_idx]

    return top_p.tolist(), top_classes

def load_category_names(path):
    with open(path, "r") as f:
        return json.load(f)
