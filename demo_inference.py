"""Script for inference
"""

import torch 
import numpy as np
from PIL import Image
from torch import nn
from preprocess import *
from pymic.transform.trans_dict import TransformDict
from pymic.net.net_dict_cls import TorchClsNetDict

class InferenceAgent():
    def __init__(self, stage):
        """
        stage: "coarse" or "fine"
        """
        self.class_num = 2 if (stage == "coarse") else 8
        self.stage = stage
        self.net_param = {"input_chns": 3, "class_num": self.class_num}
        self.net_names = ["vgg16", "resnet18", "mobilenetv2"]
        self.device = torch.device("cuda:0")
        self.networks = []
        for net_name in self.net_names:
            net = TorchClsNetDict[net_name](self.net_param).float()
            net.to(self.device)
            self.networks.append(net)

        trans_names = ["NormalizeWithMeanStd"]
        trans_param = {"task": "classification",
            "normalizewithmeanstd_channels": [0, 1, 2]}
        self.transform_list  = []
        for trans_name in trans_names:
            one_transform = TransformDict[trans_name](trans_param)
            self.transform_list.append(one_transform)

    def inference_one_image(self, img):
        """
        get the prediction of an input image
        img: an Image object after preprocessing
        """
        img_array = np.asarray(img, np.float32)
        # transpose rgb image from [H, W, C] to [C, H, W]
        img_array = np.transpose(img_array, axes = [2, 0, 1])
        img_dict  = {"image": img_array}
        if(self.stage == "coarse"):
            for trans in self.transform_list:
                img_dict = trans(img_dict)
        
        net_input = np.expand_dims(img_dict["image"], axis = 0)
        net_input = torch.from_numpy(net_input).float()
        net_input = net_input.to(self.device)
        p_list = []
        with torch.no_grad():
            for i in range(3):
                net_name = self.net_names[i]
                ckpt_name = "ckpts/{0:}/{1:}.pt".format(self.stage, net_name)
                ckpt = torch.load(ckpt_name, map_location = self.device)
                self.networks[i].load_state_dict(ckpt['model_state_dict'])
                self.networks[i].eval()
                pred = self.networks[i](net_input)
                
                if(self.stage == "coarse"):
                    prob = nn.Softmax(dim = 1)(pred).detach().cpu().numpy()[0]
                    p_list.append(prob[1:])
                else:
                    prob = nn.Sigmoid()(pred).detach().cpu().numpy()[0] 
                    p_list.append(prob)
        p_avg = np.asarray(p_list).mean(axis = 0)
        return p_avg
        
if __name__ == "__main__":
    img_name, group  ="data/data_a/20200824_1012358442.jpg", "a"  
    I  = Image.open(img_name)
    I1 = preprocess_of_one_image(I, group)
    coarse_model = InferenceAgent("coarse")
    fine_model   = InferenceAgent("fine")
    p0 = coarse_model.inference_one_image(I1)[0]
    p1 = [0.0] * 8 if p0 < 0.5 else fine_model.inference_one_image(I1)
    names = ["IgA-Kappa", "IgA-Lambda", "IgG-Kappa", "IgG-Lambda",
             "IgM-Lambda","IgM-Kappa", "Kappa", "Lambda"]
    print("probability for each pattern")
    for i in range(8):
        print(names[i].ljust(10) + ": {0:.2f}".format(p1[i]))
    

    