"""
Combined Spatial and Temporal Stream Models
"""
from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms as T
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score


# Import Custom Architectures
from Temporal.DualStreamTemporalNet import SportyTemporalStreamNet



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


class TemporalStreamFrameVideoDataset(torch.utils.data.Dataset):
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

        video_frames_dir = os.path.splitext(video_path)[0].replace(os.path.join('videos'), 'flows')
        video_frames = self.load_frames(video_frames_dir)

        frames = [self.transform(frame) if self.transform else T.ToTensor()(frame) for frame in video_frames]
        if self.stack_frames:
            frames = torch.stack(frames)  
            frames = frames.view(-1, 64, 64)  


        return frames, label
    def load_frames(self, flow_dir):
        frames = []
        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(flow_dir, f"flow_{i}_{i+1}.npy")
            flow_array = np.load(flow_file)  # shape: [2, H, W] or [H, W, 2]

            # Convert to tensor first
            flow_tensor = torch.from_numpy(flow_array).float()

            # Apply transform if defined
            if self.transform:
                flow_tensor = self.transform(flow_tensor)

            frames.append(flow_tensor)
        return frames


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    # Determine Hardware to run on
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Location of Data
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage'

    # Get transform for spatial net
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    spatial_transform = weights.transforms()
    # Get transform for temporal net
    temporal_transform = T.Compose([T.Resize((64, 64))])

    # Match images with with flows
    batch_size_spatial = 10
    frameimage_dataset_test = FrameImageDataset(root_dir=root_dir, split='test', transform=spatial_transform)
    frameimage_loader_test = DataLoader(frameimage_dataset_test, batch_size=batch_size_spatial, shuffle=False)
    

    batch_size_temporal = 1
    framevideostack_dataset_test = TemporalStreamFrameVideoDataset(root_dir=root_dir, split='test', transform=temporal_transform, stack_frames=True)
    framevideostack_loader_test = DataLoader(framevideostack_dataset_test, batch_size=batch_size_temporal, shuffle=False)
    
    
    # Load spatial and temporal stream nets
    temporalstream_model = SportyTemporalStreamNet()
    state_dict_temporal = torch.load('Temporal/temporalstream_modelstatedict_20251012-140143.pt') # Custom
    temporalstream_model.load_state_dict(state_dict_temporal)
    temporalstream_model.to(device)
    temporalstream_model.eval()

    # Define architecture for Spatial Stream
    spatialstream_model = models.efficientnet_b0()
    spatialstream_model.classifier = nn.Sequential(nn.Linear(spatialstream_model.classifier[1].in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 10)
        )
    state_dict_spatial = torch.load('Spatial/spatialstreamT_modelstatedict_20251014-003243.pt') # Load weights from training
    spatialstream_model.load_state_dict(state_dict_spatial)
    spatialstream_model.to(device)
    spatialstream_model.eval()        
    
    
    all_preds = []
    all_labels = []
    
    for i, ((video_flow_frames, label_flow),(frames, labels_imgs)) in enumerate(zip(framevideostack_loader_test,frameimage_loader_test)):
        #print(f"{i}: Video:",video_frames.shape, labels_flow, "has these images associated:", frames.shape, labels_imgs) # [batch, channels, number of frames, height, width]
        
        # Sanity check
        if torch.any(labels_imgs != label_flow):
            print("Mismatch in labels!!!")
            assert "Missmatch in labels!!! Data loading is not done properly!"

        # Move data to GPU
        video_flow_frames, label_flow = video_flow_frames.to(device), label_flow.to(device)
        frames, labels_imgs = frames.to(device), labels_imgs.to(device)

        
        with torch.no_grad():
            output_frames = spatialstream_model(frames)
            output_video_flow_frames = temporalstream_model(video_flow_frames)
        
        #print("Output of Spatial:",output_frames.argmax(1))
        #print("Output of Temporal:",output_video_flow_frames.argmax(1))
        avg_output_frames = output_frames.mean(dim=0, keepdim=True)  # [1, num_classes]
        fused_output = (avg_output_frames + output_video_flow_frames) / 2
        final_prediction = fused_output.argmax(1)

        print("True label:", label_flow)
        print("Spatial prediction:", output_frames.argmax(1))
        print("Temporal prediction:", output_video_flow_frames.argmax(1))
        print("Fused prediction:", final_prediction)
        all_preds.append(final_prediction.item())
        all_labels.append(label_flow.item())
    
    # Compute confusion matrix
    class_names = [
    'Body-WeightSquats', 'HandstandPushups', 'HandstandWalking', 'JumpingJack', 'JumpRope',
    'Lunges', 'PullUps', 'PushUps', 'TrampolineJumping', 'WallPushups']
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"DualStreamConfusionMatrix.png", dpi=300, bbox_inches='tight')
    
    # Total accuracy    
    accuracy = accuracy_score(all_labels, all_preds)

    # F1 score (macro averages across all classes)
    #f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"Total Accuracy: {accuracy:.4f}")
    #print(f"F1 Score: {f1:.4f}")

    
    print("Code finished!")

    


    


    