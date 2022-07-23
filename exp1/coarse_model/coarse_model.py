# -*- coding: utf-8 -*-
from __future__ import print_function, division
import logging
import os
import sys
from pymic.util.parse_config import *
from pymic.net_run.agent_cls import ClassificationAgent

def main():
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3 or 4, e.g.')
        print('   python coarse_model.py train config.cfg 1')
        print('   python coarse_model.py test  config.cfg 1')
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
    agent = ClassificationAgent(config, stage)
    agent.run()

if __name__ == "__main__":
    main()
    

