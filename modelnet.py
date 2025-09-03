# Complete the tasks mentioned as "TODO: Task:<>"
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO:Task 1: Import necessary libraries:
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,Dataloader

# set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# visualize point clouds for a batch
def visualize_point_cloud(batch, fig_size=(20, 20)):
    if torch.is_tensor(batch):
        batch = batch.numpy()

    fig = plt.figure(figsize=fig_size)
    for i in range(16):
        point_cloud = batch[i]
        row = i // 4
        col = i % 4
        # Create 3D subplot
        ax = fig.add_subplot(4, 4, i+1, projection='3d')

        # Extract x, y, z coordinates
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        # Plot 3D scatter
        ax.scatter(x, y, z, s=1, alpha=0.5)

        ax.set_title(f'Point Cloud {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(False)

    plt.tight_layout()
    plt.savefig('visualize.jpg', format='jpg',bbox_inches='tight')

# Point Cloud Transforms
class RandomRotation(object):
    def __call__(self, points):
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2*np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        return np.dot(points, rotation_matrix)

class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip
        
    def __call__(self, points):
        jitter = np.clip(np.random.normal(0, self.sigma, points.shape), -self.clip, self.clip)
        return points + jitter

class ToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


# ModelNet10 Dataset
# TODO:Task 2: Define ModelNet10 dataset:
class ModelNet10(Dataset):
    def __init__(self, data_path, phase, num_points, transforms):
        self.transforms=transforms
        self.num_points=num_points
        self.phase=phase

        if self.phase=='train':
            self.data=np.load(os.path.join(data_path,'points_train.npy'))
            self.labels=np.load(os.path.join(data_path,'labels_train.npy'))
        if self.phase=='test':
            self.data=np.load(os.path.join(data_path,'points_test.npy'))
            self.labels=np.load(os.path.join(data_path,'labels_test.npy'))
        print(self.data.shape, self.label.shape)

    def __getitem__(self, idx):
        pointCloud=self.data[idx][:self.num_points]
        label=self.labels[idx]
        pointCloud=self.transforms(pointCloud)
        return pointCloud,label

    def __len__(self):
        return self.data.shape[0]


# TODO:Task 3: Define PointNetClassifier model
class PointNetClassifier(nn.Module):
    def __init__(self, num_channel=3, num_classes=10):
        super(PointNetClassifier, self).__init__()


    # hint: NLL loss takes log-softmax as input
    # Use F.log_softmax across proper dimension as the final output
    # use torch max along proper dimension for maxpool
    def forward(self, x):







# TODO:Task 4:Training function
# hint:check dimension of inputs and labels while calculating loss
# use torch.transpose() and torch.squeeze() if needed
# function should return train loss and train accuracy
# print training loss after each batch for sanity check
def train(model, train_loader, optimizer, criterion, device='cpu'):








# TODO:Task 5: Evaluation function
# hint:check dimension of inputs and labels while calculating loss
# use torch.transpose() and torch.squeeze() if needed
# function should return test loss and test accuracy
def evaluate(model, val_loader, criterion, device):







def main():
    # TODO: Task 6: Initialize necessary components
    # 'Path to ModelNet10 dataset'
    data_path = './data'
    # 'Number of training epochs'
    epochs = 10
    # 'Learning rate'
    lr = 0.001

    # 'Batch size'
    batch_size = 16
    # 'Number of points in each point cloud'
    num_points = 2048
    # 'Random seed'
    seed = 42

    set_seed(seed)

    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    # TODO:Task 7: Complete train_transform and test transform code:
    # train transform: Random rotation, jitter and convert to tensor
    # test trasnfrom = convert to tensor
    train_transform = transforms.Compose([

    ])


    test_transform = 


    # TODO:Task 8 Load dataset and dataloader :
    train_dataset = ModelNet10(data_path,phase='train',num_points=2048,transforms=train_transform)

    test_dataset = ModelNet10(data_path,phase='test',num_points=2048,transforms=test_transform)

    train_loader = Dataloader(train_dataset,batch_size=batch_size,shuffle='true')

    test_loader = Dataloader(test_dataset,batch_size=batch_size,shuffle='true') 

    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    for batch, _ in train_loader:
        visualize_point_cloud(batch)
        break

    # Initialize model
    model = PointNetClassifier().to(device)
    criterion = nn.NLLLoss()

    # TODO:Task 9 optimizer
    optimizer = 


    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model!")
    
    print(f"Best validation accuracy: {best_acc:.4f}")

    # TODO:Task 10 Congrats, no more tasks

if __name__ == "__main__":
    main()