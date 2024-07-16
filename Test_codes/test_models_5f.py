import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class LeNet_relu(nn.Module):
    def __init__(self):
        super(LeNet_relu, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # Assuming input channels=1 (grayscale image)
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 32, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Assuming you have a list of Chinese labels
labels = ['Hammer', 'Air_pick', 'Excavator']

# Create label-to-index and index-to-label mappings
label_to_index = {label: idx for idx, label in enumerate(labels)}
index_to_label = {idx: label for idx, label in enumerate(labels)}


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.categories = sorted(os.listdir(root_dir))
        self.data = self.load_data()

    def load_data(self):
        data = []
        for category in self.categories:
            category_path = os.path.join(self.root_dir, category)
            file_list = [os.path.join(category_path, file) for file in os.listdir(category_path) if
                         file.endswith('.npy')]
            data.extend([(file, category) for file in file_list])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path, label = self.data[idx]
        matrix = np.load(file_path)

        if self.transform:
            matrix = self.transform(matrix)

        # Encode labels to numerical indices
        label = label_to_index[label]

        return {'matrix': matrix, 'label': label}


def check_accuracy(loader, criterion, model, device):
    model.eval()
    correct_num = 0
    total_num = 0
    total_loss = []

    with torch.no_grad():
        for batch in loader:
            matrices, targets = batch['matrix'].to(device), batch['label'].to(device)
            matrices = matrices.to(torch.float32)  # or torch.float64 based on your data type
            outputs = model(matrices)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss.append(loss.item())

            _, predicted = torch.max(outputs, 1)
            total_num += targets.size(0)
            correct_num += (predicted == targets).sum().item()

    checked_accuracy = correct_num / total_num
    checked_loss = sum(total_loss) / len(total_loss)
    # print(f'Test Accuracy: {accuracy * 100:.2f}%')
    return checked_accuracy, checked_loss

def main():
    # List of paths to the .pt files
    model_paths = ['../saved_models/alpha_01_temp_5/lenet_student_fold_1.pt',
                   '../saved_models/alpha_01_temp_5/lenet_student_fold_2.pt',
                   '../saved_models/alpha_01_temp_5/lenet_student_fold_3.pt',
                   '../saved_models/alpha_01_temp_5/lenet_student_fold_4.pt',
                   '../saved_models/alpha_01_temp_5/lenet_student_fold_5.pt']

    # Placeholder for original model's test accuracies
    test_acc_list = []
    train_acc_list = []

    for index, path in enumerate(model_paths, start=1):
        print(f"Fold {index}:")

        # Create an instance of the model
        model = LeNet_relu()

        # Load the state dictionary from the file and apply it to the model
        model.load_state_dict(torch.load(path))

        # Define a transformation to convert matrices to PyTorch tensors
        # Create instances of the dataset for training, validation, and test sets
        data_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = CustomDataset(root_dir='../../local_ds/example_samples/train', transform=data_transform)

        test_dataset = CustomDataset(root_dir='../../local_ds/example_samples/test', transform=data_transform)

        batch_size = 8

        train_loader = DataLoader(train_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

        model.to(device)


        # Test the final model on the train and test set
        train_accuracy, train_loss = check_accuracy(train_loader, criterion, model, device)
        print(f"Train Accuracy: {train_accuracy:.4f}, Test Loss: {train_loss:.4f}")

        test_accuracy, test_loss = check_accuracy(test_loader, criterion, model, device)
        print(f"Original Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

        train_accuracy, train_loss = check_accuracy(train_loader, criterion, model, device)

        print(f"Original Train Accuracy: {train_accuracy:.4f}, Test Loss: {train_loss:.4f}")

        test_accuracy, test_loss = check_accuracy(test_loader, criterion, model, device)

        print(f"Original Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

        test_acc_list.append(test_accuracy)
        train_acc_list.append(train_accuracy)
    
    # Calculate the average of the original test accuracies
    avg_test_accuracy = np.mean(test_acc_list)
    
    print(f"Average Testing Accuracy: {avg_test_accuracy:.4f}")
    
    # Calculate the average of the original train accuracies
    avg_train_accuracy = np.mean(train_acc_list)
    
    print(f"Average Training Accuracy: {avg_train_accuracy:.4f}")


if __name__ == "__main__":
    main()