import torch
import clip
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaselineModel, DomainDisentangleModel, DomainDisentangleModelDG

W1 = 0.05
W2 = 0.75
W3 = 0.04
W4 = 0.04
W5 = 0.02
W6 = 0.10

class CLIPDisentangleExperiment:

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cpu' if opt['cpu'] else 'cuda:0')
        self.model = DomainDisentangleModel()
        self.model.train()
        self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt['lr'])
        self.modelCLIP, _ = clip.load('ViT-B/32', device='cpu')
        self.modelCLIP = self.modelCLIP.to(torch.float32) # <<< dovrebbe bastare questa riga in più
        self.modelCLIP.to(self.device)
        self.modelCLIP.eval()
        for param in self.modelCLIP.parameters():
            param.requires_grad = False
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
        x, y, domain, t = data
        x = x.to(self.device)
        y = y.to(self.device)
        domain = domain.to(self.device)
        t = t.to(self.device)

        text_features = self.modelCLIP.encode_text(t)
        domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features = self.model(x, domain, flag)

        domain_loss = self.criterion(domain_logits, domain)
        if opt['DG'] == False:
          category_loss = self.criterion(category_logits, y[domain == 0])
        if opt['DG'] == True:
          category_loss = self.criterion(category_logits, y)
        domain_probs = F.softmax(domain_logits, dim=1)
        category_probs = F.softmax(category_logits, dim=1)
        domain_entropy_loss = -torch.mean(torch.sum(domain_probs * torch.log(domain_probs + 1e-8), dim=1))
        category_entropy_loss = -torch.mean(torch.sum(category_probs * torch.log(category_probs + 1e-8), dim=1))
        reconstruction_loss = self.criterion_MSEL(reconstructed_features, domain_encoded + category_encoded)
        clip_loss = self.criterion_MSEL(text_features, domain_encoded)

        loss = W1*domain_loss + W2*category_loss + W3*domain_entropy_loss + W4*category_entropy_loss + W5*reconstruction_loss + W6*clip_loss

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
            for x, y, d, t in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                d = d.to(self.device)
                t = t.to(self.device)

                domain_encoded, domain_logits, category_encoded, category_logits, reconstructed_features = self.model(x, d, flag)

                if flag == 'train':
                  if opt['DG'] == False:
                    loss += self.criterion(category_logits, y[d == 0])
                    pred = torch.argmax(category_logits, dim=-1)
                    accuracy += (pred == y[d == 0]).sum().item()
                    count += x.size(0)
                  if opt['DG'] == True:
                    loss += self.criterion(category_logits, y)
                    pred = torch.argmax(category_logits, dim=-1)
                    accuracy += (pred == y).sum().item()
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