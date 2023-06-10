import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34

def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # load image
    img_path = "./sunflower.jpg"
    assert os.path.exists(img_path), "file {} does not exit".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    #[N, C, H, W]----(batch_size channel height width)
    img =data_transform(img)# 预处理
    img = torch.unsqueeze(img, dim= 0)# 拓展第一个通道

    #read class_dict
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path,"r") as f:
        class_dict = json.load(f)

    print(class_dict)
    #creat model
    model = resnet34(num_classes= 5).to(device)

    #load model weights
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), "file '{}' does not exits".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        #predict class  由于未指定设备，PyTorch 会默认使用 CPU 进行计算
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim = 0)
        predict_cla = torch.argmax(predict).numpy()# 返回的是全连接层得到的最大参数所对应的序号
    print(output)
    print(model(img.to(device)))
    print(predict)
    print(predict_cla)
    # class_dict 字典，通过键 查找 值  并且json读出来的是 str格式，需要强制转换
    print_res = "class: {}  prob: {: .3}".format(class_dict[str(predict_cla)],predict[predict_cla].numpy())
    print(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_dict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()