import BBBLinear
from misc import FlattenLayer, ModuleWrapper
import torch.nn as nn

priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

class discriminator(ModuleWrapper):
    def __init__(self, itemCount):  # itemcount表示项目的数量
        super(discriminator, self).__init__()  # 调用父类的构造函数
        BBB_Linear = BBBLinear.BBBLinear

        self.f1 = BBB_Linear(itemCount, 1024, bias=True, priors=priors)
        self.a1 = nn.ReLU()
        self.f2 = BBB_Linear(1024, 128, bias=True, priors=priors)
        self.a2 = nn.ReLU()
        self.f3 = BBB_Linear(128, 16, bias=True, priors=priors)
        self.a3 = nn.ReLU()
        self.f4 = BBB_Linear(16, 1, bias=True, priors=priors)
        self.a4 = nn.Sigmoid()


class generator(ModuleWrapper):
    def __init__(self, itemCount):
        super(generator, self).__init__()  # 调用父类的构造函数
        BBB_Linear = BBBLinear.BBBLinear

        self.f1 = BBB_Linear(itemCount, 256, bias=True, priors=priors)  # 输入长度为100
        self.a1 = nn.ReLU()
        self.f2 = BBB_Linear(256, 512, bias=True, priors=priors)
        self.a2 = nn.ReLU()
        self.f3 = BBB_Linear(512, 1024, bias=True, priors=priors)
        self.a3 = nn.ReLU()
        self.f4 = BBB_Linear(1024, itemCount, bias=True, priors=priors)
        self.a4 = nn.Sigmoid()



