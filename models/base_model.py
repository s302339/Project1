import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet18

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.resnet18 = resnet18(pretrained=True)

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x)
        x = self.resnet18.layer4(x)
        x = self.resnet18.avgpool(x)
        if x.size(0) == 1:                  #! <<< Ho modificato solo queste due righe
            return x.squeeze().unsqueeze(0) #! <<<
        return x.squeeze()

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, 7)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.category_encoder(x)
        x = self.classifier(x)
        return x

class DomainDisentangleModel(nn.Module):
    def __init__(self):
        super(DomainDisentangleModel, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        if opt['DG'] == False:
          self.domain_classifier = nn.Linear(512, 2)
        if opt['DG'] == True:
          self.domain_classifier = nn.Linear(512, 3)

        self.category_classifier = nn.Linear(512, 7)

        self.reconstructor = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
    '''
    def forward(self, x, d, flag, branch):

      features = self.feature_extractor(x)
      if branch==0:
        domain_encoded = self.domain_encoder(features)
        domain_logits = self.domain_classifier(domain_encoded)
        return domain_logits
      if branch==1:
        category_encoded = self.category_encoder(features)
        if flag == 'train':
          category_logits = self.category_classifier(category_encoded[d == 0])
          return category_logits
        if flag == 'test':
          category_logits = self.category_classifier(category_encoded)
          return category_logits
      if branch==2:
        domain_encoded = self.domain_encoder(features)
        category_encoded = self.category_encoder(features)
        reconstructed_features = self.reconstructor(domain_encoded + category_encoded)
        return domain_encoded, category_encoded, reconstructed_features
      '''
    def forward(self, x, d, flag):
        features = self.feature_extractor(x)
        domain_encoded = self.domain_encoder(features)
        category_encoded = self.category_encoder(features)
        reconstructed_features = self.reconstructor(domain_encoded + category_encoded)
        domain_logits = self.domain_classifier(domain_encoded)
        if flag == 'train':
            category_logits = self.category_classifier(category_encoded[d == 0])
        if flag == 'test':
          category_logits = self.category_classifier(category_encoded)

        return domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features


class DomainDisentangleModelDG(nn.Module):
    def __init__(self):
        super(DomainDisentangleModelDG, self).__init__()
        self.feature_extractor = FeatureExtractor()

        self.domain_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.category_encoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.domain_classifier = nn.Linear(512, 3)
        self.category_classifier = nn.Linear(512, 7)

        self.reconstructor = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        '''
        self.image_classifier = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # 3 domains
        )

        self.mask_classifier = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 3) # 3 mask
        )
        '''

    def forward(self, x, m1, m2, m3, d):

      features = self.feature_extractor(x)
      domain_encoded = self.domain_encoder(features)
      category_encoded = self.category_encoder(features)
      reconstructed_features = self.reconstructor(domain_encoded + category_encoded)
      domain_logits = self.domain_classifier(domain_encoded)
      category_logits = self.category_classifier(category_encoded)
      '''
      image_logits = self.image_classifier(x)
      mask1_logits = self.mask_classifier(m1)
      mask2_logits = self.mask_classifier(m2)
      mask3_logits = self.mask_classifier(m3)
      '''

      return domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features #image_logits, mask1_logits, mask2_logits, mask3_logits
