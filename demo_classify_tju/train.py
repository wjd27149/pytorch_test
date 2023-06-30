import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from dataset import MyDataSet
from torchvision.models import resnet50, ResNet50_Weights

from utils import split_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from torch.utils.tensorboard import SummaryWriter
from model import Resnet50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    file = json.loads(open('config.json', 'r', encoding='utf-8').read())

    data_root_config = file["Data_Root"] 
    image_path = data_root_config["orignial_root"]  

    using_config = file["Resnet50"]
    model_name = using_config["model_name"]
    batch_size_train = using_config["dataset"]["batch_size_train"]
    batch_size_test = using_config["dataset"]["batch_size_test"]
    learning_rate = using_config["optimizer"]["lr"]
    momentum = using_config["optimizer"]["momentum"]
    n_epochs = using_config["config"]["epoch"]
    continue_train = using_config["Continue_to_train"]
    writer = SummaryWriter(f'./log/' + model_name)

    # uploader dataset
    train_images_path, train_images_label, val_images_path, val_images_label = split_data(root = image_path, val_rate= 0.2)
    
#这段代码是数据预处理中的一部分，它定义了两个数据转换的管道，分别用于训练集和验证集的图像数据预处理。
# 具体来说，这段代码中包含了四个数据转换操作：
# 1. RandomResizedCrop：随机裁剪，将图像随机裁剪成指定的大小（这里是224×224），同时还可以进行缩放和裁剪的比例随机变化，这样可以增加数据的多样性，提高模型的泛化能力。
# 2. RandomHorizontalFlip：随机水平翻转，以一定的概率对图像进行水平翻转，这样可以增加数据的多样性，提高模型的泛化能力。
# 3. ToTensor：将图像转换成张量形式，这是深度学习中常用的数据格式。
# 4. Normalize：对图像进行归一化处理，使得图像的像素值在0到1之间，并且均值为0，标准差为1。这样做的目的是为了使得不同的图像数据具有可比性，使得模型更容易学习到图像中的特征。
# 在这段代码中，对图像进行预处理的具体操作是先进行随机裁剪和水平翻转，然后将图像转换成张量形式，最后进行归一化处理。这些操作可以使得模型更好地学习到图像中的特征，并且提高模型的泛化能力。

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


# uploader dataset is uploader a picture and a root until all pic done
    train_dataset = MyDataSet(train_images_path, images_class= train_images_label, transform= data_transform["train"])
    train_num = len(train_dataset)

    nw = min([os.cpu_count(), batch_size_train if batch_size_train > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size_train, shuffle=True,
                                               num_workers=nw)

    validate_dataset = MyDataSet(images_path= val_images_path, images_class= val_images_label, transform= data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size_test, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    if model_name == 'resnet50':
        net = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in net.parameters():
            param.requires_grad = False
        class_size = 5
        in_channels = net.fc.in_features
        model.fc = nn.Linear(in_channels, class_size)
        model = model.to(device)
    elif model_name == "Resnet50":
        net = Resnet50(num_classes= 5)
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth


    # change fc layer structure
    #首先，通过 net.fc.in_features 获取了原始模型中最后一层全连接层的输入通道数（即输入特征维度)
    # 然后，通过 nn.Linear(in_channel, 5) 创建了一个新的线性层

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)
    if using_config["optimizer"]["type"] == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    best_acc = 0.0
    save_path = './resNet50.pth'
    save_pre_path = './resNet50_pre.pth'
    train_steps = len(train_loader)
    for epoch in range(n_epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            # 对历史梯度进行清零
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            # 反向传播
            loss.backward()
            # 参数的更新
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     n_epochs,
                                                                     loss)
        total_loss = running_loss / train_steps
        writer.add_scalar('loss', total_loss, epoch)
        net.eval()
        acc = 0
        predict_list = []
        for idx, batch in enumerate(validate_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs = net(images)
            _, id = torch.max(outputs.data, 1)
            predict_list.append(id.cpu())
        acc = accuracy_score(torch.cat(predict_list), val_images_label)
        recall = recall_score(torch.cat(predict_list), val_images_label, average='macro')
        precision = precision_score(torch.cat(predict_list), val_images_label, average='macro')
        f1 = f1_score(torch.cat(predict_list), val_images_label, average='macro')
        net.train()
        writer.add_scalar('acc', acc, epoch)
        writer.add_scalar('recall', recall, epoch)
        writer.add_scalar('precision', precision, epoch)
        writer.add_scalar('f1', f1, epoch)

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, total_loss, acc))

        torch.save(net.state_dict(), save_pre_path)
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()