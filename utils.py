import os
import torch
import torchvision as tv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.init as init
from torch.utils.data import ConcatDataset
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary

class GarmentDataset(Dataset):
    def __init__(self, image_dir, label_dir, classification=False, num_classes=4, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.classification = classification
        self.num_classes = num_classes
        self.image_files = [file for file in os.listdir(image_dir) if not file.startswith('Thumbs.db')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace('.jpg', '.txt'))

        color_image = Image.open(image_path).convert('RGB')
        label = self._load_label(label_path)

        if self.transform:
            color_image = self.transform(color_image)

        return color_image, label

    def _load_label(self, label_path):
        with open(label_path, 'r') as file:
            label = int(file.read().strip())
            if self.classification:
                one_hot_label = torch.zeros(self.num_classes)
                one_hot_label[label] = 1
                return one_hot_label
            else:
                return torch.tensor(label)
    

def random_split_dataset(dataset, split_size):
    random_seed = 42 #Fixed random seed to ensure same split of datasets
    torch.manual_seed(random_seed)
    num_samples = len(dataset)
    train_size = int(num_samples * split_size)
    val_size = num_samples - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return train_set, val_set

def preprocess_image(image, width, height):
    transform = tv.transforms.Compose([
        tv.transforms.Resize((width, height)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean=[0.4483, 0.4292, 0.4072],
            std=[0.2520, 0.2456, 0.2469]
        )
        ])
    transformed_image = transform(image)
    return transformed_image
        

def make_dataset(folder_path, width, height, classification, num_classes):
    image_dir = folder_path + 'dataset/images'
    if classification:
        label_dir = folder_path + 'dataset/labels_class'
    else:
        label_dir = folder_path + 'dataset/labels'

    transform = tv.transforms.Compose([
        tv.transforms.Resize((width, height)),
        tv.transforms.ToTensor()
    ])

    dataset = GarmentDataset(image_dir, label_dir, classification=classification, num_classes=num_classes, transform=transform)
    split_size = 0.8
    train_dataset, val_dataset = random_split_dataset(dataset, split_size)
    

    data_augment_horizontal_transform = tv.transforms.Compose([
        tv.transforms.Resize((width, height)),
        tv.transforms.RandomHorizontalFlip(p=1),
        tv.transforms.ToTensor()
    ])

    data_augment_vertical_transform = tv.transforms.Compose([
        tv.transforms.Resize((width, height)),
        tv.transforms.RandomVerticalFlip(p=1),
        tv.transforms.ToTensor()
    ])

    horizontal_train_dataset = deepcopy(train_dataset)
    vertical_train_dataset = deepcopy(train_dataset)

    horizontal_train_dataset.transform = data_augment_horizontal_transform
    vertical_train_dataset.transform = data_augment_vertical_transform

    train_dataset = ConcatDataset([train_dataset, horizontal_train_dataset, vertical_train_dataset])
    
    train_dataset = normalize_data(train_dataset)
    val_dataset = normalize_data(val_dataset)
    
    datasets = {'train_set': train_dataset, 'val_set': val_dataset}

    return datasets

def make_dataloaders(batch_size, datasets):
    train_loader = DataLoader(datasets['train_set'], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets['val_set'], batch_size=batch_size)
    return train_loader, val_loader

def loss_function(outputs, labels):
    MSE = nn.MSELoss()
    loss = MSE(outputs, labels)
    return loss 


def calculate_metrics(correctness_count, metrics, total):
    metrics['accuracy'] = 100*correctness_count['correct'] / total
    metrics['near1_accuracy'] = 100*correctness_count['near1'] / total
    metrics['near2_accuracy'] = 100*correctness_count['near2'] / total
    metrics['near3_accuracy'] = 100*correctness_count['near3'] / total
    return metrics

def train_step(model, optimizer, dataloader, device):
    model.train()
    criterion = nn.MSELoss()
    running_loss = 0.0
    total = 0
    metrics = {'loss': 0, 'accuracy': 0, 'near1_accuracy': 0, 'near2_accuracy': 0, 'near3_accuracy': 0}
    correctness_count = {'correct': 0, 'near1': 0, 'near2': 0, 'near3': 0}

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.unsqueeze(1).to(torch.float32)
        
        optimizer.zero_grad()
        outputs = model(images)
        rounded_output = outputs.round()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        total += labels.size(0)
        correctness_count['correct'] += (rounded_output == labels).sum().item()
        
        diff = abs(rounded_output - labels)
        for threshold in [1,2,3]:
            correctness_count[f'near{threshold}'] += sum(diff <= threshold).item()

        dataloader.set_description(f'Batch [{batch_idx+1}/{len(dataloader)}]')
        
    metrics['loss'] = running_loss / len(dataloader)
    metrics = calculate_metrics(correctness_count, metrics, total)
    
    return metrics

def train_step_classification(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    total = 0
    correct = 0
    metrics = {'loss': 0, 'accuracy': 0}
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        #labels = labels.to(torch.long)
        
        optimizer.zero_grad()
        logits = model(images)
        _, predicted = torch.max(logits, 1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += (predicted == labels.argmax(dim=1)).sum().item()
        total += labels.size(0)
        
        dataloader.set_description(f'Batch [{batch_idx+1}/{len(dataloader)}]')
    metrics['loss'] = running_loss / len(dataloader)
    metrics['accuracy'] = 100*correct / total
    return optimizer, metrics



def val_step(model, dataloader, device):
    model.eval()
    criterion = nn.MSELoss()
    running_loss = 0.0
    metrics = {'loss': 0, 'accuracy': 0, 'near1_accuracy': 0, 'near2_accuracy': 0, 'near3_accuracy': 0}
    correctness_count = {'total': 0, 'correct': 0, 'near1': 0, 'near2': 0, 'near3': 0}
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).to(torch.float32)
            
            outputs = model(images)
            rounded_output = outputs.round()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            correctness_count['total'] += labels.size(0)
            correctness_count['correct'] += (rounded_output == labels).sum().item()
            
            diff = abs(rounded_output - labels)
            for threshold in [1,2,3]:
                correctness_count[f'near{threshold}'] += sum(diff <= threshold).item()
                
            dataloader.set_description(f'Batch [{batch_idx+1}/{len(dataloader)}]')
            
    metrics['loss'] = running_loss / len(dataloader)
    metrics = calculate_metrics(correctness_count, metrics, correctness_count['total'])
    return metrics

def val_step_classification(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    metrics = {'loss': 0, 'accuracy': 0}
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            correct += (predicted == labels.argmax(dim=1)).sum().item()
            total += labels.size(0)
            
            
    metrics['loss'] = running_loss / len(dataloader)
    metrics['accuracy'] = 100*correct / total
    return metrics

def plot_loss_curves(train_losses, val_losses, name):
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(name)
    plt.show()

def plot_accuracy_curves(train_accuracies, val_accuracies, name):
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(name)
    plt.show()

def normalize_data(dataset):
    #Mean and std calculated over training set, no need to do it every time
    train_mean = [0.4483, 0.4292, 0.4072]
    train_std = [0.2520, 0.2456, 0.2469]
    normalize_transform = tv.transforms.Compose([
        tv.transforms.Normalize(mean=train_mean, std=train_std)
    ])

    dataset.transform = normalize_transform
    return dataset



def predict_single_image(model, image):
    model.eval()
    image = image.unsqueeze(0) #Adding batch dimension
    with torch.no_grad():
        output = model(image)
        rounded_output = output.round()
    return rounded_output.item()

def plot_single_image(image, predicted_label, true_label):
    plt.figure(figsize=(12, 8))
    image = tv.transforms.ToPILImage()(image)
    plt.imshow(image)

    title = f"Predicted: {predicted_label}\nTrue: {true_label}"
    title = title.replace("on ", "")  # Remove "on" and the device name
    plt.title(title)
    plt.axis('off')
    plt.show()
    
def predict_over_dataloader(model, dataloader, device, classification=False):
    model.eval()
    if classification: 
        correctness_count = {'total': 0, 'correct': 0}
        with torch.no_grad():
            for _, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                correctness_count['total'] += labels.size(0)
                correctness_count['correct'] += (predicted == labels.argmax(dim=1)).sum().item()
                
    else:      
        correctness_count = {'total': 0, 'correct': 0, 'near1': 0, 'near2': 0, 'near3': 0}
        with torch.no_grad():
            for _, (images, labels) in enumerate(dataloader):
                images = images.to(device)
                output = model(images)
                rounded_output = output.round()
                correctness_count['total'] += labels.size(0)
                correctness_count['correct'] += (rounded_output == labels).sum().item()

                diff = abs(rounded_output - labels)
                for threshold in [1,2,3]:
                    correctness_count[f'near{threshold}'] += sum(diff <= threshold).item()
                
    for key in correctness_count:
        print(f"{key}: {correctness_count[key]}")
        
    
def show_and_predict_images_from_loader(model, dataloader, device, classification): #Change name
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        with torch.no_grad():
            if classification:
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                for i in range(len(images)):
                    predicted_label = predicted[i]
                    print(labels[i])
                    true_label = torch.argmax(labels[i]).item()
                    print(true_label)
                    plot_single_image(images[i], predicted_label, true_label)
            else:
                outputs = model(images)
                rounded_output = outputs.round()
                
                for i in range(len(images)):
                    predicted_label = rounded_output[i].item() 
                    print(predicted_label)
                    true_label = labels[i].item()
                    plot_single_image(images[i], predicted_label, true_label)


            
def visualize_featureMaps(model, layer, outputs, labels, num_images_to_show): #Make to one single function
    feature_map = model.feature_maps[layer]
    fig, axs = plt.subplots(1, num_images_to_show, figsize=(15, 3))
    for i in range(num_images_to_show):
        plt.subplot(1,num_images_to_show,i+1)
        plt.imshow(feature_map[i, 0].cpu().detach(), cmap='jet')
        plt.title(f"Count: {outputs[i][0]}, Layer: {layer}")
        plt.axis('off')
        
def visualize_featureMaps_classification(model, layer, outputs, labels, num_images_to_show):
    feature_map = model.feature_maps[layer]
    fig, axs = plt.subplots(1, num_images_to_show, figsize=(15, 3))
    for i in range(num_images_to_show):
        plt.subplot(1,num_images_to_show,i+1)
        plt.imshow(feature_map[i, 0].cpu().detach(), cmap='jet')
        plt.title(f"Class: {outputs[i]}, Layer: {layer}")
        plt.axis('off')
        
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def should_early_stop(patience, epochs_without_improvement): #Improve
    if epochs_without_improvement >= patience:
        print(f"Early stopping: No improvement for {patience} epochs.")
        return True
    return False

def save_model_to_file(model, input_size, info, file_name):
    with open(file_name, 'w') as file:
        file.write(model)
        file.write("\n")
        file.write("Model information:\n")
        for key, value in info.items():
            file.write(f"{key}: {value}\n")
