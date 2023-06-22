import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaselineModel, DomainDisentangleModel, DomainDisentangleModelDG

W1 = 0.1
W2 = 0.8
W3 = 0.04
W4 = 0.04
W5 = 0.02

class DomainDisentangleExperiment:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_MSEL = torch.nn.MSELoss()

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}
        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, flag):
        x, y, domain = data
        x = x.to(self.device)
        y = y.to(self.device)
        domain = domain.to(self.device)

        '''
        domain_logits = self.model(x, domain, flag, 0)
        domain_loss = self.criterion(domain_logits, domain)
        domain_probs = F.softmax(domain_logits, dim=1)
        domain_domain2vec_loss = -torch.mean(torch.sum(domain_probs * torch.log(domain_probs + 1e-8), dim=1))
        loss_domain = W1*domain_loss + W1*domain_domain2vec_loss
        self.optimizer.zero_grad()
        loss_domain.backward()
        self.optimizer.step()

        category_logits = self.model(x, domain, flag, 1)
        category_loss = self.criterion(category_logits, y[domain == 0])
        category_probs = F.softmax(category_logits, dim=1)
        category_domain2vec_loss = -torch.mean(torch.sum(category_probs * torch.log(category_probs + 1e-8), dim=1))
        loss_category = W2*category_loss + W2*category_domain2vec_loss
        self.optimizer.zero_grad()
        loss_category.backward()
        self.optimizer.step()

        domain_encoded, category_encoded, reconstructed_features = self.model(x, domain, flag, 2)
        reconstruction_loss = W3*self.criterion_MSEL(reconstructed_features, domain_encoded + category_encoded)
        self.optimizer.zero_grad()
        reconstruction_loss.backward()
        self.optimizer.step()

        loss = W1*loss_domain + W2*loss_category + W3*reconstruction_loss
        '''

        domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features = self.model(x, domain, flag)
        domain_loss = self.criterion(domain_logits, domain)
        category_loss = self.criterion(category_logits, y[domain == 0])
        domain_probs = F.softmax(domain_logits, dim=1)
        category_probs = F.softmax(category_logits, dim=1)
        domain_domain2vec_loss = -torch.mean(torch.sum(domain_probs * torch.log(domain_probs + 1e-8), dim=1))
        category_domain2vec_loss = -torch.mean(torch.sum(category_probs * torch.log(category_probs + 1e-8), dim=1))
        reconstruction_loss = self.criterion_MSEL(reconstructed_features, domain_encoded + category_encoded)

        loss = W1*domain_loss + W2*category_loss + W3*domain_domain2vec_loss + W4*category_domain2vec_loss + W5*reconstruction_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, loader, flag):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
          for x, y, d in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            d = d.to(self.device)

            domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features = self.model(x, d, flag)

            if flag == 'train':
              loss += self.criterion(category_logits, y[d == 0])
              pred = torch.argmax(category_logits, dim=-1)
              accuracy += (pred == y[d == 0]).sum().item()
              count += x.size(0)

            if flag == 'test':
              loss += self.criterion(category_logits, y)
              pred = torch.argmax(category_logits, dim=-1)
              accuracy += (pred == y).sum().item()
              count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss

class DomainDisentangleExperimentDG:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')
        self.modelDG = DomainDisentangleModelDG()
        self.baselinemodel = BaselineModel()
        self.modelDG.train()
        self.modelDG.to(self.device)
        for param in self.modelDG.parameters():
            param.requires_grad = True
        self.baselinemodel.train()
        self.baselinemodel.to(self.device)
        for param in self.baselinemodel.parameters():
            param.requires_grad = True
        self.optimizerDG = torch.optim.Adam(self.modelDG.parameters(), lr=opt['lr'])
        self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion_MSEL = torch.nn.MSELoss()

    def save_checkpoint(self, path, iteration, best_accuracy, total_train_loss):
        checkpoint = {}
        checkpoint['iteration'] = iteration
        checkpoint['best_accuracy'] = best_accuracy
        checkpoint['total_train_loss'] = total_train_loss
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        iteration = checkpoint['iteration']
        best_accuracy = checkpoint['best_accuracy']
        total_train_loss = checkpoint['total_train_loss']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return iteration, best_accuracy, total_train_loss

    def train_iteration(self, data, flag):
        x, m1, m2, m3, y, domain = data
        x = x.to(self.device)
        m1 = m1.to(self.device)
        m2 = m2.to(self.device)
        m3 = m3.to(self.device)
        y = y.to(self.device)
        domain = domain.to(self.device)

        domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features = self.modelDG(x, m1, m2, m3, domain)
        #domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features image_logits, mask1_logits, mask2_logits, mask3_logits = self.modelDG(x, m1, m2, m3, domain)

        domain_loss = self.criterion(domain_logits, domain)
        category_loss = self.criterion(category_logits, y)
        domain_probs = F.softmax(domain_logits, dim=1)
        category_probs = F.softmax(category_logits, dim=1)
        domain_entropy_loss = -torch.mean(torch.sum(domain_probs * torch.log(domain_probs + 1e-8), dim=1))
        category_entropy_loss = -torch.mean(torch.sum(category_probs * torch.log(category_probs + 1e-8), dim=1))
        reconstruction_loss = self.criterion_MSEL(reconstructed_features, domain_encoded + category_encoded)
        '''
        loss_image = self.criterion(image_logits, domain)
        loss_mask1 = self.criterion(mask1_logits, domain)
        loss_mask2 = self.criterion(mask2_logits, domain)
        loss_mask3 = self.criterion(mask3_logits, domain)
        total_loss = loss_image + loss_mask1 + loss_mask2 + loss_mask3
        '''
        loss = W1*domain_loss + W2*category_loss + W3*domain_entropy_loss + W4*category_entropy_loss + W5*reconstruction_loss # + W6*total_loss

        self.optimizerDG.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validate(self, loader, flag):
        self.model.eval()
        accuracy = 0
        count = 0
        loss = 0
        with torch.no_grad():
          for x, m1, m2, m3, y, d in loader:
              x = x.to(self.device)
              m1 = m1.to(self.device)
              m2 = m2.to(self.device)
              m3 = m3.to(self.device)
              y = y.to(self.device)
              d = d.to(self.device)

              category_logits = self.baselinemodel(x)

              loss += self.criterion(category_logits, y)
              pred = torch.argmax(category_logits, dim=-1)
              accuracy += (pred == y).sum().item()
              count += x.size(0)

        mean_accuracy = accuracy / count
        mean_loss = loss / count
        self.model.train()
        return mean_accuracy, mean_loss