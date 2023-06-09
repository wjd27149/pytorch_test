import sys
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from vgg_model import vgg

def main():
    device = torch.device("cuda: 0" if torch.cuda.is_available() else 'cpu')
    print(f'using {device} to train')
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)) ,
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    # data_root = os.path.join(os.getcwd(), "../..",  "flower_photos")
    # 错误写法， 输出/home/liuyihan/pytorch_practise/vggnet/../../flower_photos
    data_root = os.path.abspath(os.path.join(os.getcwd(), "..", "flower_photos"))
    assert os.path.exists(data_root), "data_root {} does not exits".format(data_root)
    # 字典读取是用【】 不用()
    train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform= data_transform["train"])
    
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items()) 

    json_str = json.dumps(cla_dict, indent= 4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform= data_transform["val"])
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size = 1, num_workers = nw)
    
    model_name = "vgg16"
    net = vgg(model_name= model_name,  num_classes=5, init_weights=True)
# 顺序先把网络初始化，然后扔到device里面去训练，再初始化损失函数和参数优化器
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0001)
#   定义各种变量
    epoch = 30
    best_acc = 0.0
    save_path = './{}Net'.format(model_name)
    train_steps = len(train_loader)
    val_num = len(val_dataset)


    for i in range(epoch):
        #train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
    
    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader, file= sys.stdout)
        for step, val_data in enumerate(val_bar):
            val_images, val_labels = val_data
            val_outputs = net(val_images.to(device))
            predict_y = torch.softmax(val_outputs, dim = 1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
            
    
    print(data_root)
if __name__ == "__main__":
    main()