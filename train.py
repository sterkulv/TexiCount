import os
import torch
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import ConcatDataset
from PIL import Image
from tqdm import tqdm
from model import create_model, create_TexiCount, create_TexiCount_old
import utils
import argparse



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training Parameters")
    parser.add_argument('--optimizer', type=str, default='Adam', help="Specify optimizer (eg. Adam, SGD)")
    parser.add_argument('--epochs', type=int, default=40, help="Number of epochs to train for")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size to train with")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for optimizer")
    parser.add_argument('--path', type=str, default="./", help="Specify path to folder where files and dataset is")
    parser.add_argument('--scheduler_step', type=int, default=10, help="Step size for learning rate scheduler")
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help="Gamma value for scheduler")
    parser.add_argument('--model', type=str, default='Large', help="Specify which type of model to use")
    parser.add_argument('--filename', type=str, default='custom', help="Specify name for saving")
    parser.add_argument('--image_size', type=int, default=300, help="Specify image heigth. Width is calculated to keep the aspect ratio")
    parser.add_argument('--dropout', type=float, default=0, help="Specify how many of the weights to randomly set to 0 during training")
    parser.add_argument('--classification', type=str, default="False", help="Train the model with classifying garment type")

    args = parser.parse_args()
    

                        
    height = int(args.image_size)
    width = int(0.75*height)
    num_epochs = args.epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_loss = float('inf')
    best_acc = 0.0
    best_train_loss = float('inf')
    final_train_loss = float('inf')
    patience = 5
    epochs_without_improvement = 0

    file_name = args.filename
    print(f"Filename: {file_name}")
    best_weights_file_path = args.path + 'weights/'
    plots_file_path = args.path + 'plots/'
            
    
    if args.classification == "True":
        print("Classification")
        num_classes = 4
        
        if args.model == "Large":
            model = create_model(width=width, height=height, dropout=args.dropout, outplanes=num_classes)
        elif args.model == "Small":
            model = create_TexiCount(width=width, height=height, dropout=args.dropout, outplanes=num_classes)
        elif args.model == "Old":
            model = create_TexiCount_old(width=width, height=height, dropout=args.dropout, outplanes=num_classes)
        else: 
            print("Not a valid model, creating default model.")
            model = create_model(width=width, height=height, dropout=args.dropout, outplanes=num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        print(model)
        model.to(device)
        
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else: 
            print("Not a valid optimizer for this model! Defaulting to SGD.")
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        datasets = utils.make_dataset(args.path, width=width, height=height, classification=True, num_classes=num_classes)
        train_loader, val_loader = utils.make_dataloaders(batch_size=args.batch_size, datasets=datasets)
        
        for epoch in range(num_epochs):
            train_tqdm = tqdm(train_loader, total=len(train_loader), position=0, leave=True)
            
            optimizer, train_metrics = utils.train_step_classification(model, train_tqdm, optimizer, device=device)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_metrics['loss']:.4f}")
            print(f"Accuracy: {train_metrics['accuracy']:.4f}")
            
            scheduler.step()
            
            val_tqdm = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
            val_metrics = utils.val_step_classification(model, val_tqdm, device)
            
            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
            
            print(f'Best Previous Loss: {best_loss}')       
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_acc = val_metrics['accuracy']
                best_train_loss = train_metrics['loss']
                torch.save(model.state_dict(), best_weights_file_path + file_name)
                epochs_without_improvement = 0
                print(f'Epochs without improvement: {epochs_without_improvement}')
            else: 
                epochs_without_improvement += 1
                print(f'Epochs without improvement: {epochs_without_improvement}')


            if utils.should_early_stop(patience, epochs_without_improvement):
                final_train_loss = train_metrics['loss']
                break
    
    else: 
        print("Counting")
        num_classes = 1
        
        if args.model == "Large":
            model = create_model(width=width, height=height, dropout=args.dropout, outplanes=1)
        elif args.model == "Small":
            model = create_TexiCount(width=width, height=height, dropout=args.dropout)
        elif args.model == "Old":
            model = create_TexiCount_old(width=width, height=height, dropout=args.dropout)
        else: 
            print("Not a valid model, creating default model.")
            model = create_model(width=width, height=height, dropout=args.dropout)
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        print(model)
        model.to(device)
        
        if args.optimizer == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else: 
            print("Not a valid optimizer for this model! Defaulting to SGD.")
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
        train_near_accuracies = []
        val_near_accuracies = []
        train_accuracy = {'accuracy': 0, 'near1_accuracy': 0, 'near2_accuracy': 0, 'near3_accuracy': 0}
        val_accuracy = {'accuracy': 0, 'near1_accuracy': 0, 'near2_accuracy': 0, 'near3_accuracy': 0}    
        best_near_acc = 0.0
        datasets = utils.make_dataset(args.path, width=width, height=height, classification=False, num_classes=num_classes)
        train_loader, val_loader = utils.make_dataloaders(batch_size=args.batch_size, datasets=datasets)
            
        for epoch in range(num_epochs):
            train_tqdm = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

            train_metrics = utils.train_step(model, optimizer, train_tqdm, device=device)
            train_losses.append(train_metrics['loss'])
            train_accuracies.append(train_metrics['accuracy'])
            train_near_accuracies.append(train_metrics['near1_accuracy'])

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_metrics['loss']:.4f}")
            for key in train_metrics.keys():
                print(f"Training {key}: {train_metrics[key]:.2f}")

            scheduler.step()

            val_tqdm = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
            val_metrics = utils.val_step(model, val_tqdm, device)
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            val_near_accuracies.append(val_metrics['near1_accuracy'])

            print(f"Validation Loss: {val_metrics['loss']:.4f}")
            for key in val_accuracy.keys():
                print(f"Validation {key}: {val_metrics[key]:.2f}%")

            print(f'Best Previous Loss: {best_loss}')       
            if val_metrics['loss'] < best_loss:
                best_loss = val_metrics['loss']
                best_acc = val_metrics['accuracy']
                best_near_acc = val_metrics['near1_accuracy']
                best_train_loss = train_metrics['loss']
                torch.save(model.state_dict(), best_weights_file_path + file_name)
                epochs_without_improvement = 0
                print(f'Epochs without improvement: {epochs_without_improvement}')
            else: 
                epochs_without_improvement += 1
                print(f'Epochs without improvement: {epochs_without_improvement}')


            if utils.should_early_stop(patience, epochs_without_improvement):
                final_train_loss = train_metrics['loss']
                break
            
    
    #Save model to file
    model_info = {
        'Architecture': args.model,
        'Optimizer': args.optimizer,
        'Learning_rate': args.lr,
        'Epochs': args.epochs, 
        'Batch_size': args.batch_size,
        'Image_size': args.image_size,
        'Weight_decay': args.weight_decay,
        'Scheduler_step': args.scheduler_step,
        'Scheduler_gamma': args.scheduler_gamma,
        'Best training loss': best_train_loss,
        'Best validation loss': best_loss,
        'Best validation accuracy': best_acc,
        'Best validation near accuracy': best_near_acc,
        'Final train loss': final_train_loss,
        'Stopped after epoch': epoch      
    }
    model_file_name = file_name + '.txt'
    model_file_path = args.path + "models/"
    input_size = (3, width, height)
    utils.save_model_to_file(model, input_size, model_info, model_file_path + model_file_name)
    
    
    utils.plot_loss_curves(train_losses, val_losses, name=plots_file_path + 'Loss'+ file_name)
    utils.plot_accuracy_curves(train_accuracies, val_accuracies, name=plots_file_path + 'Acc' + file_name)
    utils.plot_accuracy_curves(train_near_accuracies, val_near_accuracies, name=plots_file_path + 'NearAcc'+file_name)