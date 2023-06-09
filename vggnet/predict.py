import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vgg_model import vgg

def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)) ,
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    img_root = "./result.jpg"
    img = Image.open(img_root)
    img = data_transform(img)
    img = torch.unsqueeze(img ,dim= 0)

    #read class_dict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path,"r") as f:
        class_dict = json.load(f)
    
    model = vgg(model_name= "vgg16", num_classes = 5).to(device)
    weight_root = "./vgg16Net.pth"
    model_lodaer = model.load_state_dict(torch.load(weight_root, map_location= device))
    model.eval()

    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cls = torch.argmax(predict)
    
    print("class: {}  prob: {}".format(class_dict[str(predict_cls)], predict[predict_cls]))