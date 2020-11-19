from torchvision.datasets import mnist
from models.alexnet import AlexNet
import time
import torch
import numpy as np
import cv2

HyperParams={
    "input_size":224,
    "input_channel":1,
    "max_epoch":5,
    "batch_size":8,
    "model_save_prefix":"./save_model",
    "cuda":False,
    "quantize":True,
}

def show_acc(target,pred):
    pass


def trainval(model,optimer,lr_sch,criterion,train_data):
    input_data = np.empty(
        (HyperParams["batch_size"], HyperParams["input_size"], HyperParams["input_size"], HyperParams["input_channel"]),
        np.float32)
    input_label = np.empty(HyperParams["batch_size"], np.int)

    for epoch in range(HyperParams["max_epoch"]):
        model.train()
        for idx,(data,label) in enumerate(zip(train_data.train_data,train_data.train_labels)):

            data = data.numpy()
            data = cv2.resize(data,(224,224))[...,np.newaxis]
            # data = np.concatenate([data for _ in range(HyperParams["input_channel"])], -1)
            data = data.astype(np.float32)/255
            input_data[idx%HyperParams["batch_size"],...] = data
            input_label[idx%HyperParams["batch_size"]] = label.numpy()
            if idx %HyperParams["batch_size"] ==(HyperParams["batch_size"]-1):
                input_data_tensor = torch.from_numpy(input_data)
                input_label_tensor = torch.from_numpy(input_label)
                if HyperParams["cuda"]:
                    input_data_tensor = input_data_tensor.cuda()
                    input_label_tensor=input_label_tensor.cuda()
                logit = model(input_data_tensor.permute([0,3,1,2]))
                optimer.zero_grad()
                loss = criterion(logit,input_label_tensor)
                loss.backward()
                optimer.step()
                if HyperParams["cuda"]:
                    loss_value = loss.cpu().detach().numpy()
                else:
                    loss_value = loss.detach().numpy()
                print("|epoch:%d|loss value:%.3f|lr:%f|"%(epoch,loss_value,optimer.param_groups[0]["lr"]))
        torch.save(model.state_dict(),HyperParams["model_save_prefix"]+"_E%d.snap"%epoch)
        lr_sch.step()
        model.eval()
        for idx,(data,label) in enumerate(zip(train_data.test_data,train_data.test_labels)):
            data = data.numpy()
            data = cv2.resize(data,(224,224))[...,np.newaxis]
            # data = np.concatenate([data for _ in range(HyperParams["input_channel"])], -1)
            data = data.astype(np.float32)/255
            input_data[idx%HyperParams["batch_size"],...] = data
            input_label[idx%HyperParams["batch_size"]] = label.numpy()
            if idx %HyperParams["batch_size"] ==(HyperParams["batch_size"]-1):
                input_data_tensor = torch.from_numpy(input_data)
                input_label_tensor = torch.from_numpy(input_label)
                if HyperParams["cuda"]:
                    input_data_tensor = input_data_tensor.cuda()
                    input_label_tensor=input_label_tensor.cuda()
                logit = model(input_data_tensor.permute([0,3,1,2]))
                pred = -torch.log_softmax(logit,-1)
                pred_cls = torch.argmax(pred,-1)

            # print('')

def test(model,test_data):
    P=0
    N=0

    input_data = np.empty(
        (HyperParams["batch_size"], HyperParams["input_size"], HyperParams["input_size"], HyperParams["input_channel"]),
        np.float32)
    input_label = np.empty(HyperParams["batch_size"], np.int)
    model.eval()
    start_time = time.time()
    for idx, (data, label) in enumerate(zip(test_data.test_data, test_data.test_labels)):
        data = data.numpy()
        data = cv2.resize(data, (224, 224))[..., np.newaxis]
        # data = np.concatenate([data for _ in range(HyperParams["input_channel"])], -1)
        data = data.astype(np.float32) / 255
        input_data[idx % HyperParams["batch_size"], ...] = data
        input_label[idx % HyperParams["batch_size"]] = label.numpy()
        if idx % HyperParams["batch_size"] == (HyperParams["batch_size"] - 1):
            input_data_tensor = torch.from_numpy(input_data)
            input_label_tensor = torch.from_numpy(input_label)
            if HyperParams["cuda"]:
                input_data_tensor = input_data_tensor.cuda()
                input_label_tensor = input_label_tensor.cuda()
            logit = model(input_data_tensor.permute([0, 3, 1, 2]))
            pred_cls = torch.argmax(logit, -1)


            P+=(pred_cls == input_label_tensor).sum().cpu().detach().numpy()
            N+=HyperParams["batch_size"]
        if idx % 500 == 499:
            print("|acc:%f|use time:%s|"%(float(P/N),str(time.time()-start_time)))
            start_time = time.time()


            # print('')
if __name__ == '__main__':
    train_data = mnist.MNIST("./mnist_data")
    model = AlexNet(10)
    if HyperParams["cuda"]:
        model = model.cuda()
    optimer = torch.optim.Adam(params=[{"params": model.parameters()}], lr=0.004)
    lr_sch = torch.optim.lr_scheduler.MultiStepLR(optimer, [1, 2, 3, 4], 0.1)
    criterion = torch.nn.CrossEntropyLoss()
    static_params = torch.load("./%s_E%d.snap"%(HyperParams["model_save_prefix"],4))
    model.load_state_dict(static_params)
    # trainval(model,optimer,lr_sch,criterion,train_data)
    if HyperParams["quantize"]:
        model = torch.quantization.quantize_dynamic(model)
    torch.save(model.state_dict(),"./quantize_mode.snap")
    test(model,train_data)
