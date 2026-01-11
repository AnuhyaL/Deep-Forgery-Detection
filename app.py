import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm
from facenet_pytorch import MTCNN
from flask import Flask, request, render_template
import io
import base64
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

class_names = ['fake', 'real']

mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('inception_resnet_v2', pretrained=False).to(device)
model.load_state_dict(torch.load('best_model_epoch_3.pt', map_location=device))
model.eval()

def smooth_grad(input_image, model, predicted_class, sigma=0.20, n_samples=25):
    mean = torch.zeros_like(input_image)
    std = torch.ones_like(input_image) * sigma

    total_gradients = torch.zeros_like(input_image)

    for _ in range(n_samples):
        noise = torch.normal(mean=mean, std=std).to(input_image.device)
        perturbed_input = input_image + noise
        perturbed_input.requires_grad = True

        output = model(perturbed_input)
        loss = -output[:, predicted_class]  

        model.zero_grad()
        loss.backward()

        gradients = perturbed_input.grad.detach()
        total_gradients += gradients

    avg_gradients = total_gradients / n_samples
    return avg_gradients


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file:
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            faces = mtcnn(image)
            if faces is None:
                return render_template('index.html', message='Please upload a valid image containing a face')
            
            input_image = test_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_image)

            predicted_class = torch.argmax(output, dim=1).item()
            confidence = F.softmax(output, dim=1)[0][predicted_class].item()
            predicted_class_name = class_names[predicted_class]

            if predicted_class_name == 'real':
                folder_name = 'Real'
            else:
                folder_name = 'Fake'

            # Create directory if it doesn't exist
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Save the uploaded image into the respective folder
            image.save(os.path.join(folder_name, file.filename))

            input_image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            input_image_np = (input_image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)  # Reverse normalization

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(input_image_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            smooth_gradients = smooth_grad(input_image, model, predicted_class)
            smooth_gradients = smooth_gradients.squeeze().permute(1, 2, 0).cpu().numpy()

            smooth_gradients_normalized = (smooth_gradients - np.min(smooth_gradients)) / (np.max(smooth_gradients) - np.min(smooth_gradients))

            axes[1].imshow(smooth_gradients_normalized)
            axes[1].set_title('SmoothGrad')
            axes[1].axis('off')

            axes[2].imshow(input_image_np)
            axes[2].imshow(smooth_gradients_normalized, alpha=0.5, cmap='jet')
            axes[2].set_title(f'Predicted: {predicted_class_name} ({confidence:.2f})')
            axes[2].axis('off')

            img_data = io.BytesIO()
            plt.savefig(img_data, format='png')
            img_data.seek(0)
            img_base64 = base64.b64encode(img_data.getvalue()).decode()

            return render_template('result.html', img_data=img_base64)

    return render_template('index.html')


@app.route('/images')
def images():
    # Assuming your images are in the 'static' folder
    image_names = ['training_acc.png', 'training_loss.png', 'confusion.png', 'comparis.png']
    return render_template('images.html', image_names=image_names)

if __name__ == '__main__':
    app.run(debug=True)
