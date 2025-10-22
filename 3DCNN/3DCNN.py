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


# 3D CNN, this is an inflated version of the Sporty2DNet used for early fusion 
class Sporty3DNet(nn.Module):
    def __init__(self, T=10):
        super(Sporty3DNet, self).__init__()
        # Input will be 3 cubes, one for each color channel. Each of these input channels has the dimension of TxHxW
        # Hence, input: 3xTxHxW
        self.convolutional = nn.Sequential(
            nn.Conv3d(in_channels=3,
                      kernel_size=(3,7,7),
                      stride=(1,2,2),
                      padding=(1,3,3),
                      out_channels=3*T*2), # Out 10x32x32x60 RF_spatial:7 RF_temporal:3
            nn.BatchNorm3d(3*T*2),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=3*T*2,
                      kernel_size=(3,3,3),
                      stride=(2,2,2),
                      padding=(3,1,1),
                      padding_mode='reflect', # To prevent eliminating the temporal dimension
                      out_channels=2*3*T*2), # Out 7x16x16x120 RF_spatial:11 RF_temporal: 5
            nn.BatchNorm3d(2*3*T*2),
            nn.ReLU(),

            nn.Conv3d(in_channels=2*3*T*2,
                      kernel_size=(3,3,3),
                      stride=1,
                      padding=(1,1,1),
                      padding_mode='reflect',
                      out_channels=2*3*T*2), # Out 7x16x16x120 RF_spatial:19 RF_temporal:9
            nn.BatchNorm3d(2*3*T*2),
            nn.ReLU(),

            nn.MaxPool3d(kernel_size=2,
                         stride=(1,2,2)), # Out: 6x8x8x120 RF_spatial:23 RF_temporal:11

            nn.Conv3d(in_channels=2*3*T*2,
                      kernel_size=(3,3,3),
                      stride=1,
                      padding=1,
                      padding_mode='reflect',
                      out_channels=2*2*3*T*2), # Out 6x8x8x240 RF_spatial:39 RF_temporal:15
            nn.BatchNorm3d(2*2*3*T*2),
            nn.ReLU(),
            
            #nn.Dropout3d(0.5),
            
            nn.Conv3d(in_channels=2*2*3*T*2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect',
                      out_channels=2*2*3*T*2), # Out 6x8x8x240 RF_spatial:50 RF_temporal:19
            nn.BatchNorm3d(2*2*3*T*2),
            nn.ReLU(),
            
            #nn.Dropout3d(0.5),

            nn.Conv3d(in_channels=2*2*3*T*2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect',
                      out_channels=2*2*3*T*2), # Out 6*8x8x240 RF_spatial:66 RF_temporal:23
            nn.BatchNorm3d(2*2*3*T*2),
            nn.ReLU(),
        )
            
        self.fully_connected = nn.Sequential(
                nn.Linear(6*8*8*240, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 10))
    
    def forward(self, x):
        x = self.convolutional(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        x = self.fully_connected(x)
        return x

def train_3DCNN(model, optimizer, train_loader, test_loader, trainset, testset, num_epochs=10):
    
    device = next(model.parameters()).device  # Automatically detect model's device
    print("Training will be carried out on:", device)

    loss_fun = nn.CrossEntropyLoss()  # Instantiate once
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
            #Forward pass your image through the network
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
        print(f"Epoch: {epoch}; Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
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
        #print(video_meta['label'])
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
    transform = T.Compose([T.Resize((64, 64)), T.ToTensor(),T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    """
    frameimage_dataset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)
    framevideostack_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    framevideolist_dataset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=False)

    frameimage_loader = DataLoader(frameimage_dataset, batch_size=8, shuffle=False)
    framevideostack_loader = DataLoader(framevideostack_dataset, batch_size=8, shuffle=False)
    framevideolist_loader = DataLoader(framevideolist_dataset, batch_size=8, shuffle=False)
    """
    batch_size = 8
    framevideostack_dataset_train = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    framevideostack_dataset_val = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)
    framevideostack_loader_train = DataLoader(framevideostack_dataset_train, batch_size=batch_size, shuffle=True)
    framevideostack_loader_val = DataLoader(framevideostack_dataset_val, batch_size=batch_size, shuffle=True)

    #for frames, labels in frameimage_loader:
    #     print(frames.shape, labels.shape) # [batch, channels, height, width]

    #for video_frames, labels in framevideolist_loader:
    #     print(45*'-')
    #     for frame in video_frames: # loop through number of frames
    #         print(frame.shape, labels.shape)# [batch, channels, height, width]


    for video_frames, labels in framevideostack_loader_val:
        print(video_frames.shape, labels.shape) # [batch, channels, number of frames, height, width]
        break
    

    
    # Create instance of early fusion net and optimizer
    sporty_3D = Sporty3DNet()
    optimizer = torch.optim.Adam(sporty_3D.parameters(), lr=0.00001, weight_decay=1e-4)

    # Send model to GPU
    sporty_3D.to(device)

    trainset = framevideostack_dataset_train
    testset = framevideostack_dataset_val  # or use a separate test split
    
    # Train and test
    num_epochs = 15
    out = train_3DCNN(sporty_3D, optimizer,
                    train_loader=framevideostack_loader_train,
                    test_loader=framevideostack_loader_val,
                    trainset=trainset, testset=testset,
                    num_epochs=num_epochs)


    # Plotting and saving files:
    epochs_lin = np.arange(1, num_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Sporty3DNet')

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
    plt.savefig(f'3DCNN_traininghist_{timestamp}.png')
    #plt.show()

    # Save model
    torch.save(sporty_3D.state_dict(), f'3DCNN_modelstatedict_{timestamp}.pt')
    torch.save(sporty_3D, f'3DCNN_model_{timestamp}.pt')

    # Save training history
    with open(f'3DCNN_traininghist_{timestamp}.json', 'w') as f:
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

    print("Code finished!")
    """

    


    


    