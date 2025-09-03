# PointNet Implementation for ModelNet10 Classification
1. Project Synopsis
This project provides a PyTorch implementation of a simplified PointNet-like architecture for 3D object classification. The script is designed to train and evaluate the model on the ModelNet10 dataset, which consists of 3D point cloud data for 10 object categories. The implementation covers data loading, preprocessing, augmentation, model definition, training, and validation.

2. Technical Specifications
Framework: PyTorch

Dataset: ModelNet10 (pre-processed as .npy files)

Core Architecture: PointNet (simplified)

Task: 3D Point Cloud Classification

Input Data Format: Point Clouds of size (N, 2048, 3), where N is the number of samples.

3. System Requirements
3.1. Libraries and Dependencies
The following Python libraries are required to run the script. They can be installed via pip.

Bash

pip install torch torchvision numpy matplotlib
3.2. Data Requirements
The script requires the ModelNet10 dataset to be pre-processed into four specific NumPy files. These files must be placed within a data/ directory located in the project's root folder.

data/points_train.npy

data/labels_train.npy

data/points_test.npy

data/labels_test.npy

4. Installation and Execution
4.1. Setup Instructions
Clone the repository or place the modelnet.py script in a new directory.

Create the data directory:

Bash

mkdir data
Place the dataset files into the newly created ./data/ directory as specified in section 3.2.

Install the dependencies as specified in section 3.1.

4.2. Running the Script
Execute the main Python script from the command line:

Bash

python modelnet.py
4.3. Expected Output
Upon execution, the script will:

Print the device being used (e.g., CPU).

Print the number of training and testing samples.

Generate an image file named visualize.jpg showing a sample batch of point clouds.

For each epoch, print the training loss, training accuracy, validation loss, and validation accuracy.

Save the model state dictionary with the highest validation accuracy to a file named best_model.pth.

5. Codebase Description
The modelnet.py script contains all the necessary components for the project.

Dataset Class (ModelNet10): A custom torch.utils.data.Dataset class for loading and handling the .npy data files.

