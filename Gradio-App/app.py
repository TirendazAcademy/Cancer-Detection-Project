import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import gradio as gr

title = "Cancer Detection"
description = "Image classification with histopathologic images"
article = "<p style='text-align: center'><a href='https://github.com/TirendazAcademy'>Github Repo</a></p>"

# The model architecture
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrain_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.pretrain_model.eval()
        for param in self.pretrain_model.parameters():
            param.requires_grad = False       
        self.pretrain_model.fc = nn.Sequential(
            nn.Linear(self.pretrain_model.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024,2)
        )
    def forward(self, input):
        output=self.pretrain_model(input)
        return output
        
model = ImageClassifier()
model.load_state_dict(torch.load('model-data_comet-torch-model.pth'))

def predict(inp):
    image_transform = transforms.Compose([ transforms.Resize(size=(224,224)), transforms.ToTensor()])
    labels = ['normal', 'cancer']
    inp = image_transform(inp).unsqueeze(dim=0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp))
        confidences = {labels[i]: float(prediction.squeeze()[i]) for i in range(len(labels))}    
    return confidences
    
gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             title=title,
             description=description,
             article=article,
             examples=['image-1.jpg', 'image-2.jpg']).launch()