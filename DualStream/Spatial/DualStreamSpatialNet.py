from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialStreamConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SpatialStreamConvNet, self).__init__()

        # conv1: 7x7x96, stride 2, norm., pool 2x2
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # conv2: 5x5x256, stride 2, norm., pool 2x2
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # conv3: 3x3x512, stride 1
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # conv4: 3x3x512, stride 1, pool 2x2
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # conv5: 3x3x512, stride 1
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc6 = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Adjust input size based on input resolution
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc7 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)  # [B, 96, H/2, W/2]
        x = self.conv2(x)  # [B, 256, H/4, W/4]
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.classifier(x)
        return x

class BigSportySpatialStreamNet(nn.Module):
    def __init__(self):
        super(BigSportySpatialStreamNet, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(7*7*256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
class SportySpatialStreamNet(nn.Module):
    def __init__(self):
        super(SportySpatialStreamNet, self).__init__()
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      out_channels=32), # Out 32x32x32 RF:7
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      out_channels=32), # Out 16x16x64 RF:11
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #nn.Dropout2d(0.5),

            nn.Conv2d(in_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      out_channels=32), # Out 16x16x64 RF:19
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2,
                         stride=2), # Out: 8*8*64 RF:23

            nn.Conv2d(in_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      out_channels=32), # Out 8x8x64 RF:39
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Dropout2d(0.2),
            
            nn.Conv2d(in_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      out_channels=32), # Out 8x8x64 RF:55
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Dropout2d(0.2),

            nn.Conv2d(in_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      out_channels=32), # Out 8x8x64 RF:71
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
            
        self.fully_connected = nn.Sequential(
                nn.Linear(8*8*32, 128),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(128, 10))
    
    def forward(self, x):
        x = self.convolutional(x)
        # Reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x
    
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


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir='/dtu/datasets1/02516/ucf101_noleakage', split='train', transform=None, stack_frames=True):
        video_glob = os.path.join(root_dir, 'videos', split, '*', '*.avi')
        self.video_paths = sorted(glob(video_glob))
        
        csv_path = os.path.join(root_dir, 'metadata', f'{split}.csv')
        self.df = pd.read_csv(csv_path)
        
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames
        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)
    
    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_meta = self._get_meta('video_name', video_name)
        print(video_meta['label'])
        label = video_meta['label'].item()

        video_frames_dir = os.path.splitext(video_path)[0].replace(os.path.join('videos'), 'frames')
        video_frames = self.load_frames(video_frames_dir)

        frames = [self.transform(frame) if self.transform else T.ToTensor()(frame) for frame in video_frames]
        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label
    
    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames
    


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

    transform = T.Compose([T.Resize((64, 64)),
                           T.ToTensor(),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
                           #T.RandomHorizontalFlip(0.5)
                            ]
                           )

    """
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=False)

    frameimage_loader = DataLoader(frameimage_dataset, batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset, batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset, batch_size=8, shuffle=False)
    """
    batch_size = 32
    frameimage_dataset_train = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
    frameimage_dataset_val = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
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
    sporty_spatialstream = SportySpatialStreamNet()
    optimizer = torch.optim.Adam(sporty_spatialstream.parameters(), lr=0.00001, weight_decay=1e-3)

    # Send model to GPU
    sporty_spatialstream.to(device)

    trainset = frameimage_dataset_train
    print("Len of trainset:",len(frameimage_dataset_train))
    testset = frameimage_dataset_val  # or use a separate test split
    
    # Train and test
    num_epochs = 20

    out = train_spatialstream(sporty_spatialstream, optimizer,
                    train_loader=frameimage_loader_train,
                    test_loader=frameimage_loader_val,
                    trainset=trainset, testset=testset,
                    num_epochs=num_epochs)


    # Plotting and saving files:
    epochs_lin = np.arange(1, num_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('SportySpatialStreamNet')

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
    plt.savefig(f'spatialstream_traininghist_{timestamp}.png')
    #plt.show()

    # Save model
    torch.save(sporty_spatialstream.state_dict(), f'spatialstream_modelstatedict_{timestamp}.pt')
    torch.save(sporty_spatialstream, f'spatialstream_model_{timestamp}.pt')

    # Save training history
    with open(f'spatialstream_traininghist_{timestamp}.json', 'w') as f:
        json.dump(out, f)

    """
    # Visualization of prediction
    class_names = [
    'Body-WeightSquats', 'HandstandPushups', 'HandstandWalking', 'JumpingJack', 'JumpRope',
    'Lunges', 'PullUps', 'PushUps', 'TrampolineJumping', 'WallPushups'
    ]

    # Define a visualization-only transform (no normalization)
    transform_vis = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])

    # Create a visualization dataset instance
    framevideostack_dataset_vis = EarlyFusionFrameVideoDataset(
        root_dir=root_dir, split='val', transform=transform_vis, stack_frames=True
    )

    # Load one sample from visualization dataset
    frames, label = framevideostack_dataset_vis[0]  # frames: [30, 64, 64]

    # Convert back to 10 RGB frames
    T_frames = 10
    frames = frames.view(T_frames, 3, 64, 64)  # [10, 3, 64, 64]

    # Pick a frame to visualize (e.g., middle one)
    frame_to_show = TF.to_pil_image(frames[T_frames // 2])  # Convert tensor to PIL

    # Run model prediction using normalized input
    sporty_early_fusion.eval()
    with torch.no_grad():
        input_tensor = frames.view(-1, 64, 64).unsqueeze(0).to(device)  # [1, 30, 64, 64]
        output = sporty_early_fusion(input_tensor)
        predicted_class = output.argmax(1).item()
        predicted_label = class_names[predicted_class]
        true_label = class_names[label]

    # Save figure to disk and show
    fig = plt.figure()
    plt.imshow(frame_to_show)
    plt.title(f"Predicted: {predicted_label} | True: {true_label}")
    plt.axis('off')
    fig.savefig(f"earlyfusion_prediction_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    """

    print("Code finished!")

    


    


    