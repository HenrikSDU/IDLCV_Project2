from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms as T
import torchvision.models as models
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import time


def train_spatialstream(model, optimizer, train_loader, test_loader, trainset, testset, num_epochs=10):
    
    device = next(model.parameters()).device  # Automatically detect model's device
    print("Training of spatial stream net is starting and will be carried out on:", device)

    loss_fun = nn.CrossEntropyLoss(label_smoothing=0.1)  # Instantiate once
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': []}
  
    for epoch in range(num_epochs):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        for minibatch_no, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
        out_dict['train_acc'].append(train_correct/len(trainset))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Epoch of spatial stream: {epoch}; Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
    return out_dict


class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='/dtu/datasets1/02516/ucf101_noleakage', split='train', transform=None):
        frame_glob = os.path.join(root_dir, 'frames', split, '*', '*', '*.jpg')
        self.frame_paths = sorted(glob(frame_glob))
        
        csv_path = os.path.join(root_dir, 'metadata', f'{split}.csv')
        self.df = pd.read_csv(csv_path)
        
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = os.path.basename(os.path.dirname(frame_path))  # robust extraction
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        frame = Image.open(frame_path).convert("RGB")
        frame = self.transform(frame) if self.transform else T.ToTensor()(frame)

        return frame, label



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    # Determine Hardware to run on
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    base_transform = weights.transforms()
    transform = T.Compose([
    weights.transforms(),  # base transform
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ])

    """
    transform = T.Compose([T.Resize((224, 224)),
                           T.ToTensor(),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
                           #T.RandomHorizontalFlip(0.5)
                            ]
                           )

    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=False)

    frameimage_loader = DataLoader(frameimage_dataset, batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset, batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset, batch_size=8, shuffle=False)
    """
    batch_size = 32
    frameimage_dataset_train = FrameImageDataset(root_dir=root_dir, split='train', transform=base_transform)
    frameimage_dataset_val = FrameImageDataset(root_dir=root_dir, split='val', transform=base_transform)
    frameimage_loader_train = DataLoader(frameimage_dataset_train, batch_size=batch_size, shuffle=True)
    frameimage_loader_val = DataLoader(frameimage_dataset_val, batch_size=batch_size, shuffle=True)

    """
    for frames, labels in frameimage_loader:
        print(frames.shape, labels.shape) # [batch, channels, height, width]
        break
    for video_frames, labels in framevideolist_loader:
        print(45*'-')
        for frame in video_frames: # loop through number of frames
            print(frame.shape, labels.shape)# [batch, channels, height, width]
            break
        break
    """


    #for video_frames, labels in framevideostack_loader_val:
     #   print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
     #   break
    #for video_frames, labels in framevideostack_loader_train:
     #   print(video_frames.shape)  # Should be [batch, 30, 64, 64]
      #  break

    
    # Create instance of spatial stream net and optimizer
    sporty_spatialstream = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    sporty_spatialstream.classifier = nn.Sequential(nn.Linear(sporty_spatialstream.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )
    # Freeze feature extraction layers
    for param in sporty_spatialstream.features.parameters():
        param.requires_grad = True
    
    #for name, param in sporty_spatialstream.features.named_parameters():
     #   if "6" in name or "7" in name:  # EfficientNet-B0 has 8 blocks
      #      param.requires_grad = True


    optimizer = torch.optim.Adam(sporty_spatialstream.parameters(), lr=1e-4, weight_decay=1e-3)
        #[
                               # {'params': sporty_spatialstream.features.parameters(), 'lr': 1e-5},
                   #             {'params': sporty_spatialstream.classifier.parameters(), 'lr': 1e-4}
                            #]
                          

    # Send model to GPU
    sporty_spatialstream.to(device)

    trainset = frameimage_dataset_train
    print("Len of trainset:",len(frameimage_dataset_train))
    testset = frameimage_dataset_val  # or use a separate test split
    
    # Train and test
    num_epochs = 5
    out = train_spatialstream(sporty_spatialstream, optimizer,
                    train_loader=frameimage_loader_train,
                    test_loader=frameimage_loader_val,
                    trainset=trainset, testset=testset,
                    num_epochs=num_epochs)


    # Plotting and saving files:
    epochs_lin = np.arange(1, num_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('SportySpatialStreamNet based on EfficientNetB0')

    # Plotting loss
    ax1.plot(epochs_lin, out['test_loss'], label='Test Loss')
    ax1.plot(epochs_lin, out['train_loss'], label='Train Loss')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plotting accuracy
    ax2.plot(epochs_lin, out['test_acc'], label='Test Accuracy')
    ax2.plot(epochs_lin, out['train_acc'], label='Train Accuracy')
    ax2.set_title('Accuracy over Epochs')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit suptitle

    # Generate timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save plot
    plt.savefig(f'spatialstreamT_traininghist_{timestamp}.png')
    #plt.show()

    # Save model
    torch.save(sporty_spatialstream.state_dict(), f'spatialstreamT_modelstatedict_{timestamp}.pt')
    torch.save(sporty_spatialstream, f'spatialstreamT_model_{timestamp}.pt')

    # Save training history
    with open(f'spatialstreamT_traininghist_{timestamp}.json', 'w') as f:
        json.dump(out, f)


    print("Code finished!")

    


    


    