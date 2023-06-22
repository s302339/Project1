import os
import cv2
import clip
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.manifold import TSNE
import torchvision.transforms as T
from torch.autograd import forward_ad
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoaderimport os
from parse_args import parse_arguments
from load_data import build_splits_baseline, build_splits_domain_disentangle, build_splits_clip_disentangle
from experiments.baseline import BaselineExperiment
from experiments.domain_disentangle import DomainDisentangleExperiment
from experiments.clip_disentangle import CLIPDisentangleExperiment

def setup_experiment(opt):

    source_CLIP_dictionary = ''
    target_CLIP_dictionary = ''

    if opt['experiment'] == 'baseline':
        experiment = BaselineExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_baseline(opt)

    elif opt['experiment'] == 'domain_disentangle':
        experiment = DomainDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader = build_splits_domain_disentangle(opt)

    elif opt['experiment'] == 'clip_disentangle':
        experiment = CLIPDisentangleExperiment(opt)
        train_loader, validation_loader, test_loader, source_CLIP_dictionary, target_CLIP_dictionary = build_splits_clip_disentangle(opt)

    else:
        raise ValueError('Experiment not yet supported.')

    return experiment, train_loader, validation_loader, test_loader, source_CLIP_dictionary, target_CLIP_dictionary

def main(opt):
    experiment, train_loader, validation_loader, test_loader, source_CLIP_dictionary, target_CLIP_dictionary = setup_experiment(opt)

    flag = 'train'

    if not opt['test']:
        iteration = 0
        best_accuracy = 0
        total_train_loss = 0

        if os.path.exists(f'{opt["output_path"]}/last_checkpoint.pth'):
            iteration, best_accuracy, total_train_loss = experiment.load_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth')
        else:
            logging.info(opt)

        while iteration < opt['max_iterations']:
            for data in train_loader:

              total_train_loss += experiment.train_iteration(data, flag)

              if iteration % opt['print_every'] == 0:
                  logging.info(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')
                  print(f'[TRAIN - {iteration}] Loss: {total_train_loss / (iteration + 1)}')  # AGGIUNTO PERCHE' NON MI CREA IL FILE DI LOG

              if iteration % opt['validate_every'] == 0:
                  # Run validation
                  val_accuracy, val_loss = experiment.validate(validation_loader, flag)
                  logging.info(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')
                  print(f'[VAL - {iteration}] Loss: {val_loss} | Accuracy: {(100 * val_accuracy):.2f}')  # AGGIUNTO PERCHE NON MI CREA IL FILE DI LOG
                  if val_accuracy > best_accuracy:
                      best_accuracy = val_accuracy
                      experiment.save_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth', iteration, best_accuracy, total_train_loss)
                  experiment.save_checkpoint(f'{opt["output_path"]}/last_checkpoint.pth', iteration, best_accuracy, total_train_loss)

              iteration += 1
              if iteration > opt['max_iterations']:
                  break

    experiment.load_checkpoint(f'{opt["output_path"]}/best_checkpoint.pth')
    flag = 'test'
    test_accuracy, _ = experiment.validate(test_loader, flag)
    logging.info(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')
    print(f'[TEST] Accuracy: {(100 * test_accuracy):.2f}')  # AGGIUNTO PERCHE' NON MI CREA IL FILE DI LOG

if __name__ == '__main__':

    opt = parse_arguments()
    '''
    opt = {
    'experiment': 'domain_disentangle',   # 'baseline', 'domain_disentangle', 'clip_disentangle'
    'target_domain': 'cartoon',           # 'cartoon', 'sketch', 'photo'
    'lr': 0.001,
    'max_iterations': 10000,
    'batch_size': 32,
    'num_workers': 1,
    'print_every': 50,
    'validate_every': 100,
    'output_path': '.',
    'data_path': 'Project1/data/PACS',
    'cpu': False,
    'test': False,
    'DG': False
    }
    opt['output_path'] = f'{opt["output_path"]}/record/{opt["experiment"]}_{opt["target_domain"]}'
    '''
    
    # Setup output directories
    os.makedirs(opt['output_path'], exist_ok=True)

    # Setup logger
    logging.basicConfig(filename=f'{opt["output_path"]}/log.txt', format='%(message)s', level=logging.INFO, filemode='a')

    main(opt)