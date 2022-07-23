# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys
import numpy as np 
from torchvision import datasets, models, transforms
from pymic.util.parse_config import *
from pymic.io.nifty_dataset import ClassificationDataset
from pymic.net_run.agent_cls import ClassificationAgent


class PhrosisDataset(ClassificationDataset):
    def __init__(self, root_dir, csv_file, modal_num = 1, class_num = 2, 
            with_label = False, transform=None):
        super(PhrosisDataset, self).__init__(root_dir, 
            csv_file, modal_num, with_label, transform)
        self.class_num = class_num
        print("class number for ClassificationDataset", self.class_num)

    def __getlabel__(self, idx):
        label = self.csv_items.iloc[idx, 1:9]
        label = np.asarray(list(label))
        return label
   
class PhrosisAgent(ClassificationAgent):
    def __init__(self, config, stage = 'train'):
        super(PhrosisAgent, self).__init__(config, stage)

    def get_stage_dataset_from_config(self, stage):
        assert(stage in ['train', 'valid', 'test'])
        root_dir  = self.config['dataset']['root_dir']
        modal_num = self.config['dataset']['modal_num']

        if(stage == "train"):
            transform_names = self.config['dataset']['train_transform']
        elif(stage == "valid"):
            transform_names = self.config['dataset']['valid_transform']
        elif(stage == "test"):
            transform_names = self.config['dataset']['test_transform']
        else:
            raise ValueError("Incorrect value for stage: {0:}".format(stage))
        self.transform_list  = []
        if(transform_names is None or len(transform_names) == 0):
            data_transform = None 
        else:
            transform_param = self.config['dataset']
            transform_param['task'] = 'classification' 
            for name in transform_names:
                if(name not in self.transform_dict):
                    raise(ValueError("Undefined transform {0:}".format(name))) 
                one_transform = self.transform_dict[name](transform_param)
                self.transform_list.append(one_transform)
            data_transform = transforms.Compose(self.transform_list)

        csv_file = self.config['dataset'].get(stage + '_csv', None)
        class_num = self.config['network']['class_num']
        dataset  = PhrosisDataset(root_dir=root_dir,
                                csv_file  = csv_file,
                                modal_num = modal_num,
                                class_num = class_num,
                                with_label= not (stage == 'test'),
                                transform = data_transform )
        return dataset

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3 or 4, e.g.')
        print('   python fine_model.py train config.cfg 1')
        print('   python fine_model.py test  config.cfg 1')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    fold     = int(sys.argv[3])
    config   = parse_config(cfg_file)
    config   = synchronize_config(config)
    fi = "f{0:}".format(fold)
    if(stage == "train"):
        config['dataset']["train_csv"] = config['dataset']["train_csv"].replace(
            "fk", fi)
        config['dataset']["valid_csv"] = config['dataset']["valid_csv"].replace(
            "fk", fi)
        config["training"]["ckpt_save_dir"] = config['training']["ckpt_save_dir"].replace(
            "fk", fi) 
        log_dir  = config['training']['ckpt_save_dir']
    else:
        config["training"]["ckpt_save_dir"] = config['training']["ckpt_save_dir"].replace(
            "fk", fi) 
        config['testing']["output_csv"] = config['testing']["output_csv"].replace(
            "fk", fi)
        log_dir = "/".join(config['testing']["output_csv"].split("/")[:-1])
        print(log_dir)
    if(not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    logging.basicConfig(filename=log_dir+"/log.txt", level=logging.INFO,
                        format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging_config(config)
    agent = PhrosisAgent(config, stage)
    agent.run()

if __name__ == "__main__":
    main()
