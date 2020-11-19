import torch
from torch import nn
class Classifier(nn.Module):
    def __init__(self, in_size, in_ch):
        super(Classifier, self).__init__()
        self.layer1 = nn.Conv2d(in_ch,6,5,1,2,groups=2,bias=True)
        self.fc = nn.Linear(6 * in_size * in_size, 1)

        self.layer1.register_backward_hook(self.grad_hook)
    @staticmethod
    def grad_hook(md, grad_in, grad_out):
        # grad_in 包含： grad_bias, grad_x, grad_w 三者的梯度： (delta_bias, delta_x, delta_w)
        # grad_out 是md整体的梯度，也等于grad_bias
        print('========= register_backward_hook input:======== ')
        for x in grad_in:
            if x == None:
                print("None")
            else:
                print(x.size())
                print(x)
            print("---------------------------------------------")
        print('========= register_backward_hook output:======== ')
        for x in grad_out:
            if x == None:
                continue
            print(x.size())
            print(x)
            print("---------------------------------------------")

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def backward(self,x,grad_out):

        return x


def print_grad(grad):
    print('========= register_hook output:======== ')
    print(grad.size())
    print(grad)




torch.random.manual_seed(1000)

if __name__ == '__main__':
    in_size, in_ch = 5, 4
    x = torch.randn(1, in_ch, in_size, in_size)
    model = Classifier(in_size, in_ch)
    y_hat = model(x)
    y_gt = torch.Tensor([[1.5]])
    crt = nn.MSELoss()
    print(y_hat)
    print('=======================')
    identity = []

    loss = crt(y_hat, y_gt)
    loss.backward()

    # print(x)