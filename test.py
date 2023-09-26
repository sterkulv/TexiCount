import model
import utils
import torch
import argparse
import torchvision as tv
from torch.utils.data import DataLoader
from tqdm import tqdm





if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parameters to run test script")
    parser.add_argument('--model', type=str, default="Large", help="Specify which model to use")
    parser.add_argument('--path', type=str, default='./dataset', help='Specify path to folder of images to predict count')
    parser.add_argument('--weights', type=str, default='./weights/best_weights', help='Specify the weights you want to use')
    parser.add_argument('--show_images', type=str, default="true", help="Show images with prediction or not")
    parser.add_argument('--classification', type=str, default="False", help="Classify garment type, not count")
    parser.add_argument('--image_size', type=int, default=300, help="Specify size of image")
    
    args = parser.parse_args()
    
    folder_path = args.path
    
    height = int(args.image_size)
    width = int(0.75*height)
    classification = False
    
    if args.classification == "True":
        classification = True
        print("Classification")
        num_classes = 4
    
        if args.model == "Large":
            model = model.create_model(width=width, height=height, dropout=0, outplanes=num_classes)
        elif args.model == "Small":
            model = model.create_TexiCount(width=width, height=height, dropout=0, outplanes=num_classes)
        elif args.model == "Old":
            model = model.create_TexiCount_old(width=width, height=height, dropout=0, outplanes=num_classes)
    else:
        print("Count")
        num_classes = 1
    
        if args.model == "Large":
            model = model.create_model(width=width, height=height, dropout=0, outplanes=num_classes)
        elif args.model == "Small":
            model = model.create_TexiCount(width=width, height=height, dropout=0, outplanes=num_classes)
        elif args.model == "Old":
            model = model.create_TexiCount_old(width=width, height=height, dropout=0, outplanes=num_classes)
    
        
    device = 'cpu'
    model.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
    
    image_dir = folder_path + '/images'
    label_dir = folder_path + '/labels'

    transform = tv.transforms.Compose([
        tv.transforms.Resize((width, height)),
        tv.transforms.ToTensor()
    ])

    dataset = utils.GarmentDataset(image_dir, label_dir, classification=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1)
    dataloader_tqdm = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    if args.show_images == "true":
        print(classification)
        utils.show_and_predict_images_from_loader(model, dataloader_tqdm, device, classification=classification)
    else: 
        utils.predict_over_dataloader(model, dataloader_tqdm, device, classification=classification)