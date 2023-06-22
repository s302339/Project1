from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.transforms import CenterCrop, Resize
import torchvision.transforms.functional as F
import random
import json
import clip


CATEGORIES = {
    'dog': 0,
    'elephant': 1,
    'giraffe': 2,
    'guitar': 3,
    'horse': 4,
    'house': 5,
    'person': 6,
}

DOMAIN = {
    'art_painting': 0,
    'cartoon': 1,
    'sketch': 2,
    'photo': 3,
}

class PACSDatasetBaseline(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y

class PACSDatasetDisentangle(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        return x, y, domain

mask_ratio = 0.2
def create_random_mask(image_shape, mask_ratio):
    mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
    mask_pixels = int(mask_ratio * np.prod(image_shape[:2]))
    mask_indices = np.random.choice(np.prod(image_shape[:2]), mask_pixels, replace=False)
    mask_indices = np.unravel_index(mask_indices, image_shape[:2])
    mask_pixels = [(x, y) for x, y in zip(mask_indices[1], mask_indices[0])]
    for pixel in mask_pixels:
        mask.putpixel(pixel, 255)
    return mask

class PACSDatasetDisentangleDG(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        image = Image.open(img_path)
        mask1 = create_random_mask(image.size, mask_ratio)
        masked_image1 = Image.fromarray(np.array(image) * np.array(mask1)[:, :, np.newaxis])
        m1 = self.transform(masked_image1.convert('RGB'))
        mask2 = create_random_mask(image.size, mask_ratio)
        masked_image2 = Image.fromarray(np.array(image) * np.array(mask2)[:, :, np.newaxis])
        m2 = self.transform(masked_image2.convert('RGB'))
        mask3 = create_random_mask(image.size, mask_ratio)
        masked_image3 = Image.fromarray(np.array(image) * np.array(mask3)[:, :, np.newaxis])
        m3 = self.transform(masked_image3.convert('RGB'))

        return x, y, domain #m1, m2, m3, y, domain

class PACSDatasetClipDisentangle(Dataset):
    def __init__(self, examples, CLIP_list, transform):
        self.examples = examples
        self.transform = transform
        self.CLIP_list = CLIP_list

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        img_path, y, domain = self.examples[index]
        x = self.transform(Image.open(img_path).convert('RGB'))
        text = 'no description'

        for d in self.CLIP_list:
          if img_path in d:
            text = d[img_path]
            text = clip.tokenize(text).squeeze()
        if text == 'no description':
          text = clip.tokenize(text).squeeze()

        return x, y, domain, text


def read_lines(data_path, domain_name):
    examples = {}
    with open(f'{data_path}/{domain_name}.txt') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()[0].split('/')
        category_name = line[3]
        category_idx = CATEGORIES[category_name]
        image_name = line[4]
        image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
        if category_idx not in examples.keys():
            examples[category_idx] = [image_path]
        else:
            examples[category_idx].append(image_path)
    return examples

def read_lines2(data_path, domain_name):
    list_ = []
    examples_dictionary = {}
    examples_list=[]

    with open(f'{data_path}/descriptions.TXT') as f:
        lines = f.readlines()
    domain = ''
    for i, line in enumerate(lines):
        list_.append(line.strip())
        if (i == 1) or ((i - 1) % 11 == 0) :
          line = line.strip().split()[0].split('/')
          domain = line[0]
          if line[0] == domain_name:
            list_ = list_[-2:]
            category_name = line[1]
            category_idx = CATEGORIES[category_name]
            image_name = line[2]
            image_path = f'{data_path}/kfold/{domain_name}/{category_name}/{image_name}'
            examples_dictionary['image_name'] = image_path
        if (i+1) %11 == 0:
          if domain == domain_name:
            examples_dictionary['descriptions'] = list_[2:]
            examples_list.append(examples_dictionary)
            list_=[]
            examples_dictionary={}

    with open(f'{data_path}/groupe1AML.txt') as f:
        lines = f.readlines()
    domain = ''
    for i, line in enumerate(lines):
        list_diction = json.loads(line)
        for diction in list_diction:
          for key, value in diction.items():
            if key == "image_name":
              domain = value.strip().split()[0].split('/')[4]
              if domain == domain_name:
                examples_list.append(diction)

    text_dict = {}
    text_list = []
    for d in examples_list:
      newkey = d['image_name']
      descr = d['descriptions']
      newvalue = ''
      for ds in descr:
        newvalue += ds
      text_dict[newkey] = newvalue
      text_list.append(text_dict)
      text_dict = {}
    return text_list


def build_splits_baseline(opt):
    source_domain = 'art_painting'
    target_domain = opt['target_domain']

    source_examples = read_lines(opt['data_path'], source_domain)
    target_examples = read_lines(opt['data_path'], target_domain)

    source_category_examples = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
    source_total_examples = sum(source_category_examples.values())
    source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_examples.items()}

    val_split_length = source_total_examples * 0.2

    train_examples = []
    val_examples = []
    test_examples = []

    for category_idx, examples_list in source_examples.items():
        split_idx = round(source_category_ratios[category_idx] * val_split_length)
        for i, example in enumerate(examples_list):
            if i > split_idx:
                train_examples.append([example, category_idx])
            else:
                val_examples.append([example, category_idx])

    for category_idx, examples_list in target_examples.items():
        for example in examples_list:
            test_examples.append([example, category_idx])

    normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = T.Compose([
        T.Resize(256),
        T.RandAugment(3, 15),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    eval_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    train_loader = DataLoader(PACSDatasetBaseline(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
    val_loader = DataLoader(PACSDatasetBaseline(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
    test_loader = DataLoader(PACSDatasetBaseline(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def build_splits_domain_disentangle(opt):

    if opt['DG'] == False:
      source_domain = 'art_painting'
      target_domain = opt['target_domain']

      source_examples = read_lines(opt['data_path'], source_domain)
      target_examples = read_lines(opt['data_path'], target_domain)

      source_category_examples = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
      source_total_examples = sum(source_category_examples.values())
      source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_examples.items()}

      target_category_examples = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
      target_total_examples = sum(target_category_examples.values())
      target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_examples.items()}

      source_val_split_length = source_total_examples * 0.2
      train_examples_S = []
      val_examples_S = []

      for category_idx, examples_list in source_examples.items():
          split_idx = round(source_category_ratios[category_idx] * source_val_split_length)
          for i, example in enumerate(examples_list):
              if i > split_idx:
                  train_examples_S.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[source_domain]' con semplicemente uno 0
              else:
                  val_examples_S.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[source_domain]' con semplicemente uno 0

      target_val_split_length = target_total_examples * 0.2
      train_examples_T = []
      val_examples_T = []
      test_examples = []

      for category_idx, examples_list in target_examples.items():
          split_idx = round(target_category_ratios[category_idx] * target_val_split_length)
          for i, example in enumerate(examples_list):
              if i > split_idx:
                  train_examples_T.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1
              else:
                  val_examples_T.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1

      for category_idx, examples_list in target_examples.items():
          for example in examples_list:
              test_examples.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1

      train_examples = train_examples_S + train_examples_T
      random.shuffle(train_examples)
      train_examples = train_examples[:len(train_examples)//2]
      val_examples = val_examples_S + val_examples_T
      random.shuffle(val_examples)
      val_examples = val_examples[:len(val_examples)//2]

      normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      train_transform = T.Compose([
          T.Resize(256),
          T.RandAugment(3, 15),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])

      eval_transform = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])

      train_loader = DataLoader(PACSDatasetDisentangle(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
      val_loader = DataLoader(PACSDatasetDisentangle(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
      test_loader = DataLoader(PACSDatasetDisentangle(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    if opt['DG'] == True:
      target_domain = opt['target_domain']
      source_domains = ['art_painting', 'cartoon', 'sketch', 'photo']
      source_domains.remove(target_domain)

      source_total_examples = 0
      train_examples_ = []
      val_examples_ = []
      for domain in source_domains:
        source_examples = read_lines(opt['data_path'], domain)
        source_category_example = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
        source_total_examples += sum(source_category_example.values())
        source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_example.items()}
        for category_idx, examples_list in source_examples.items():
          for i, example in enumerate(examples_list):
            train_examples_.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[domain]' con semplicemente uno 0
            val_examples_.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[domain]' con semplicemente uno 0

      source_train_split_length = source_total_examples * 0.8
      random.shuffle(train_examples_)
      train_examples = train_examples_[:int(source_train_split_length)]
      train_examples = train_examples[:len(train_examples)//3]
      source_val_split_length = source_total_examples * 0.2
      random.shuffle(val_examples_)
      val_examples = val_examples_[:int(source_val_split_length)]
      val_examples = val_examples[:len(val_examples)//3]

      test_examples = []
      target_examples = read_lines(opt['data_path'], target_domain)
      for category_idx, examples_list in target_examples.items():
          for example in examples_list:
              test_examples.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1

      normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      train_transform = T.Compose([
          T.Resize(256),
          T.RandAugment(3, 15),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])

      eval_transform = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])

      train_loader = DataLoader(PACSDatasetDisentangleDG(train_examples, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
      val_loader = DataLoader(PACSDatasetDisentangleDG(val_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
      test_loader = DataLoader(PACSDatasetDisentangleDG(test_examples, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader


def build_splits_clip_disentangle(opt):

    if opt['DG'] == False:
      source_domain = 'art_painting'
      target_domain = opt['target_domain']

      source_examples = read_lines(opt['data_path'], source_domain)
      target_examples = read_lines(opt['data_path'], target_domain)
      source_CLIP = read_lines2(opt['data_path'], source_domain)
      target_CLIP = read_lines2(opt['data_path'], target_domain)
      CLIP_list = source_CLIP + target_CLIP

      source_category_examples = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
      source_total_examples = sum(source_category_examples.values())
      source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_examples.items()}

      target_category_examples = {category_idx: len(examples_list) for category_idx, examples_list in target_examples.items()}
      target_total_examples = sum(target_category_examples.values())
      target_category_ratios = {category_idx: c / target_total_examples for category_idx, c in target_category_examples.items()}

      source_val_split_length = source_total_examples * 0.2
      train_examples_S = []
      val_examples_S = []

      for category_idx, examples_list in source_examples.items():
          split_idx = round(source_category_ratios[category_idx] * source_val_split_length)
          for i, example in enumerate(examples_list):
              if i > split_idx:
                  train_examples_S.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[source_domain]' con semplicemente uno 0
              else:
                  val_examples_S.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[source_domain]' con semplicemente uno 0

      target_val_split_length = target_total_examples * 0.2
      train_examples_T = []
      val_examples_T = []
      test_examples = []

      for category_idx, examples_list in target_examples.items():
          split_idx = round(target_category_ratios[category_idx] * target_val_split_length)
          for i, example in enumerate(examples_list):
              if i > split_idx:
                  train_examples_T.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1
              else:
                  val_examples_T.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1

      for category_idx, examples_list in target_examples.items():
          for example in examples_list:
              test_examples.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1

      train_examples = train_examples_S + train_examples_T
      random.shuffle(train_examples)
      val_examples = val_examples_S + val_examples_T
      random.shuffle(val_examples)

      normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      train_transform = T.Compose([
          T.Resize(256),
          T.RandAugment(3, 15),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])

      eval_transform = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])

      train_loader = DataLoader(PACSDatasetClipDisentangle(train_examples, CLIP_list, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
      val_loader = DataLoader(PACSDatasetClipDisentangle(val_examples, CLIP_list, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
      test_loader = DataLoader(PACSDatasetClipDisentangle(test_examples, CLIP_list, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)


    if opt['DG'] == True:
      target_domain = opt['target_domain']
      source_domains = ['art_painting', 'cartoon', 'sketch', 'photo']
      source_domains.remove(target_domain)

      source_total_examples = 0
      train_examples_ = []
      val_examples_ = []
      for domain in source_domains:
        source_examples = read_lines(opt['data_path'], domain)
        source_CLIP = read_lines2(opt['data_path'], source_domains)
        source_category_example = {category_idx: len(examples_list) for category_idx, examples_list in source_examples.items()}
        source_total_examples += sum(source_category_example.values())
        source_category_ratios = {category_idx: c / source_total_examples for category_idx, c in source_category_example.items()}
        for category_idx, examples_list in source_examples.items():
          for i, example in enumerate(examples_list):
            train_examples_.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[domain]' con semplicemente uno 0
            val_examples_.append([example, category_idx, 0]) #! <<< Modificare quel 'DOMAIN[domain]' con semplicemente uno 0

      source_train_split_length = source_total_examples * 0.8
      random.shuffle(train_examples_)
      train_examples = train_examples_[:int(source_train_split_length)]
      train_examples = train_examples[:len(train_examples)//3]
      source_val_split_length = source_total_examples * 0.2
      random.shuffle(val_examples_)
      val_examples = val_examples_[:int(source_val_split_length)]
      val_examples = val_examples[:len(val_examples)//3]

      test_examples = []
      target_examples = read_lines(opt['data_path'], target_domain)
      target_CLIP = read_lines2(opt['data_path'], target_domain)
      for category_idx, examples_list in target_examples.items():
          for example in examples_list:
              test_examples.append([example, category_idx, 1]) #! <<< Modificare quel 'DOMAIN[target_domain]' con semplicemente un 1

      CLIP_list = source_CLIP + target_CLIP
      normalize = T.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

      train_transform = T.Compose([
          T.Resize(256),
          T.RandAugment(3, 15),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])

      eval_transform = T.Compose([
          T.Resize(256),
          T.CenterCrop(224),
          T.ToTensor(),
          normalize
      ])


      train_loader = DataLoader(PACSDatasetClipDisentangle(train_examples, CLIP_list, train_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=True)
      val_loader = DataLoader(PACSDatasetClipDisentangle(val_examples, CLIP_list, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)
      test_loader = DataLoader(PACSDatasetClipDisentangle(test_examples, CLIP_list, eval_transform), batch_size=opt['batch_size'], num_workers=opt['num_workers'], shuffle=False)

    return train_loader, val_loader, test_loader, source_CLIP, target_CLIP