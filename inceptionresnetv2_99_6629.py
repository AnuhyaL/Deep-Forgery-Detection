

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


base_dir = '/kaggle/input/deepfake-dataset/real_vs_fake/real-vs-fake/'
train_dataset = ImageFolder(root=base_dir+'/train', transform=train_transform)
valid_dataset = ImageFolder(root=base_dir+'/valid', transform=valid_transform)
test_dataset = ImageFolder(root=base_dir+'/test', transform=valid_transform)


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes

print("Number of classes:", len(class_names))
print("Class names:")
for name in class_names:
    print(name)

import timm
import torch


model = timm.create_model('inception_resnet_v2', pretrained=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


checkpoint_dir = 'Weights'
os.makedirs(checkpoint_dir, exist_ok=True)

learning_rate = 0.001
num_epochs = 10

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_accuracy = 0.0

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_predictions / total_samples

    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_loss /= len(valid_loader)
    val_accuracy = 100 * correct_predictions / total_samples

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch + 1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved best model checkpoint at {checkpoint_path}")

model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in tqdm(valid_loader, desc='Validation'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_predictions)
class_report = classification_report(all_labels, all_predictions)

precision_per_class = precision_score(all_labels, all_predictions, average=None)
recall_per_class = recall_score(all_labels, all_predictions, average=None)
f1_score_per_class = f1_score(all_labels, all_predictions, average=None)

print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
print("\nPrecision per Class:", precision_per_class)
print("Recall per Class:", recall_per_class)
print("F1-score per Class:", f1_score_per_class)


plt.figure(figsize=(8, 40))


plt.subplot(7, 1, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(7, 1, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training and Validation Accuracy')


plt.subplot(7, 1, 3)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('True Label')
plt.ylabel('Predicted Label')


plt.subplot(7, 1, 4)
plt.bar(range(len(precision_per_class)), precision_per_class, color='blue', alpha=0.7, label='Precision')
plt.bar(range(len(recall_per_class)), recall_per_class, color='orange', alpha=0.7, label='Recall')
plt.bar(range(len(f1_score_per_class)), f1_score_per_class, color='green', alpha=0.7, label='F1-score')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Precision, Recall, F1-score per Class')
plt.legend()


plt.subplot(7, 1, 5)
sns.boxplot(data=[train_losses, val_losses], palette=['blue', 'orange'])
plt.xticks([0, 1], ['Train', 'Validation'])
plt.xlabel('Dataset')
plt.ylabel('Loss')
plt.title('Box Plot of Loss')


plt.subplot(7, 1, 6)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')


plt.subplot(7, 1, 7)
sns.histplot(train_accuracies, bins=10, kde=True, color='blue')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Distribution of Training Accuracy')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
plt.bar(['Train', 'Validation'], [train_losses[-1], val_losses[-1]], color=['blue', 'orange'])
plt.xlabel('Dataset')
plt.ylabel('Loss')
plt.title('Comparison of Loss between Train and Validation Sets')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(train_accuracies, bins=10, kde=True, color='blue')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Distribution of Training Accuracy')

plt.tight_layout()
plt.show()

print("Training complete.")





