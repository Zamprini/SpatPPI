import torch
import torch.nn as nn
import copy
import os
import torch.nn.functional as F
import numpy as np
import warnings
import argparse
import random
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from sklearn.model_selection import train_test_split 
from prettytable import PrettyTable
from torch.utils.data import DataLoader 
from tqdm import tqdm
from time import time
from SpatPPI_Model import *
from configs import *


class Normalize(nn.Module): 
    def __init__(self, features, epsilon=1e-6):
        super(Normalize, self).__init__()
        self.gain = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon

    def forward(self, x, dim=-1):
        mu = x.mean(dim, keepdim=True)
        sigma = torch.sqrt(x.var(dim, keepdim=True) + self.epsilon)
        gain = self.gain
        bias = self.bias
        # Reshape
        if dim != -1:
            shape = [1] * len(mu.size())
            shape[dim] = self.gain.size()[0]
            gain = gain.view(shape)
            bias = bias.view(shape)
        return gain * (x - mu) / (sigma + self.epsilon) + bias

class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(parameters, d_model, step_each_epoch):
    warmup_epoch = 5 
    warmup = warmup_epoch * step_each_epoch
    top_lr = 0.0004
    factor = top_lr / (d_model ** (-0.5) * min(warmup ** (-0.5), warmup * warmup ** (-1.5)))

    return NoamOpt(
        d_model, factor, warmup, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    )

def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class EdgeFeatures(nn.Module):
    def __init__(self, edge_dim,top_k):
        super(EdgeFeatures, self).__init__()
        self.top_k = top_k
        self.norm_edges = Normalize(edge_dim) 
    
    def _dist(self, X, mask, eps=1E-6):
        #Calculate pairwise euclidean distances
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2) # mask [B, L] => mask_2D [B, L, L]
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2) # X coordinate matrix [B, L, 3]   dX coordinate difference matrix [B, L, L, 3]
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps) # distance matrix [B, L, L]

       #Mark k-nearest neighbors(including the node itself)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, self.top_k, dim=-1, largest=False) # [B, L, k]  D_neighbors is the specific distance value (smallest to largest)，

        return D_neighbors, E_idx    
    
    def _quaternions(self, R):
        """ Converting 3D rotation matrices to 4-element numbers
            R [B,L,K,3,3]
            Q [B,L,K,4]
        """ 
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1) 

        return Q

    def _orientations(self, X, O, E_idx, eps=1e-6):
        """ Compute the local coordinates and splice them with the rotation matrix elements
            X coordinate matrix [B, L, 3] ; O coordinate system matrix [B, L, 3, 3]
            O_features [B, L, K, 7]
        """
        O_neighbors = gather_coordinate(O, E_idx) # [B, L, K, 3, 3]
        X_neighbors = gather_nodes(X, E_idx) # [B, L, K, 3]
        # Center coordinates of k neighboring points in the base coordinate system
        dX = X_neighbors - X.unsqueeze(-2) # [B, L, K, 3]
        dX = dX.float()
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1) # [B, L, K, 3]
        dU = F.normalize(dU, dim=-1)
        # Rotation matrix of k neighboring points in the base coordinate system
        R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
        Q = self._quaternions(R) # [B, L, K, 4]
        # Spliced Composition Edge Features
        O_features = torch.cat((dU,Q), dim=-1) # [B, L, K, 7]

        return O_features

    def forward(self, X, O, mask): 
        """ 
            X coordinate matrix [B, L, 3] ; O coordinate system matrix [B, L, 3, 3]
            E_idx Number of the corresponding neighbor node [B, L, K]
        """ 
        # data enhancement
        #if self.training and self.augment_eps > 0:
            #X = X + self.augment_eps * torch.randn_like(X)
        # Constructing a K-nearest neighbor graph
        D_neighbors, E_idx = self._dist(X, mask)
        # Calculating edge properties
        O_features = self._orientations(X, O, E_idx)
        E = self.norm_edges(O_features)

        return E, E_idx

class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, opt_da=None, discriminator=None,
                 experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.alpha = alpha
        self.n_class = config["DECODER"]["BINARY"]

        self.random_layer = False
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = model
        self.best_epoch = None
        self.best_auprc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.EdgeFeatures = EdgeFeatures(config["PROTEIN"]["EDGE_DIM"], config["ATTENTION"]["NEIGHBORS"])


    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auprc >= self.best_auprc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auprc = auprc
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
            if self.current_epoch % 5 == 0:
                Auroc, Auprc, F1, _, _, _, _, _, _ = self.test(dataloader="test")
                print('Test at Epoch ' + str(self.current_epoch) + " AUROC " + str(Auroc) + " AUPRC " + str(Auprc) +
                      " F1-score " + str(F1))
            if self.current_epoch % 15 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
                state = {
                        "train_epoch_loss": self.train_loss_epoch,
                        "val_epoch_loss": self.val_loss_epoch,
                        "test_metrics": self.test_metrics,
                        "config": self.config
                            }
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def _compute_entropy_weights(self, logits):
        entropy = entropy_logits(logits)
        entropy = ReverseLayerF.apply(entropy, self.alpha)
        entropy_w = 1.0 + torch.exp(-entropy)
        return entropy_w

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (p1, p2, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            X1,O1,V1,mask1 = p1['X'],p1['O'],p1['V'],p1['mask']
            X2,O2,V2,mask2 = p2['X'],p2['O'],p2['V'],p2['mask']
            E1,E_idx1 = self.EdgeFeatures(X1, O1, mask1)
            E2,E_idx2 = self.EdgeFeatures(X2, O2, mask2)
            V1,E1,E_idx1,mask1 = V1.to(self.device),E1.to(self.device),E_idx1.to(self.device),mask1.to(self.device)
            V2,E2,E_idx2,mask2 = V2.to(self.device),E2.to(self.device),E_idx2.to(self.device),mask2.to(self.device)
            labels = labels.float().to(self.device)
            self.optim.zero_grad()
            emb1, emb2, f, score = self.model(V1, E1, E_idx1, mask1, V2, E2, E_idx2, mask2)
            if self.n_class == 1: 
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (p1, p2, labels) in enumerate(data_loader):
                X1,O1,V1,mask1 = p1['X'],p1['O'],p1['V'],p1['mask']
                X2,O2,V2,mask2 = p2['X'],p2['O'],p2['V'],p2['mask']
                E1,E_idx1 = self.EdgeFeatures(X1, O1, mask1)
                E2,E_idx2 = self.EdgeFeatures(X2, O2, mask2)
                V1,E1,E_idx1,mask1 = V1.to(self.device),E1.to(self.device),E_idx1.to(self.device),mask1.to(self.device)
                V2,E2,E_idx2,mask2 = V2.to(self.device),E2.to(self.device),E_idx2.to(self.device),mask2.to(self.device)
                labels = labels.float().to(self.device)
                if dataloader == "val":
                    emb1, emb2, f, score = self.model(V1, E1, E_idx1, mask1, V2, E2, E_idx2, mask2)
                elif dataloader == "test":
                    emb1, emb2, score, att = self.best_model(V1, E1, E_idx1, mask1, V2, E2, E_idx2, mask2, mode="eval")
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches
        if dataloader == "test":
            y_pred_file = os.path.join(self.output_dir, "y_pred.txt")
            with open(y_pred_file, 'w') as fp: 
                for idx in range(len(y_pred)):   
                     fp.write(f"{y_pred[idx]:.2f}\t{y_label[idx]:.2f}\n") 
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
            if self.experiment:
                self.experiment.log_curve("test_roc curve", fpr, tpr)
                self.experiment.log_curve("test_pr curve", recall, prec)
            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1
        else:
            return auroc, auprc, test_loss



warnings.filterwarnings("ignore")

def main(data_folder, train_filename, test_filename):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}\n")

    # Load configuration
    cfg = get_cfg_defaults()

    # Set paths
    train_path = os.path.join(data_folder, train_filename)
    test_path = os.path.join(data_folder, test_filename)

    # Read CSV files
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Set random seed
    random_state = cfg.SOLVER.SEED

    # Split the dataset
    train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=random_state)

    # Create datasets
    train_dataset = DataToGraph(train_df.index.values, df_train, os.path.join(data_folder, 'Tensor/'))
    val_dataset = DataToGraph(val_df.index.values, df_train, os.path.join(data_folder, 'Tensor/'))
    test_dataset = DataToGraph(df_test.index.values, df_test, os.path.join(data_folder, 'Tensor/'))

    # DataLoader parameters
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS, 'drop_last': True}

    # Create DataLoaders
    training_generator = DataLoader(train_dataset, **params)
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    # Initialize model and optimizer
    model = SpatPPI(**cfg).to(device)
    opt = get_std_opt(model.parameters(), cfg.PROTEIN.EMBEDDING_DIM[0], len(training_generator))

    # Initialize and run trainer
    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None, discriminator=None, experiment=None, **cfg)
    result = trainer.train()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Train Model")

    # Command-line arguments
    parser.add_argument('--data_folder', type=str, required=True, help="Path to the data folder containing CSV files and tensors")
    parser.add_argument('--train_filename', default="HuRI-IDP-Train.csv", type=str, help='The filename of the train dataset under data_folder(csv file)')
    parser.add_argument('--test_filename', default="HuRI-IDP-TestA.csv", type=str, help='The filename of the test dataset under data_folder(csv file')

    # Parse arguments
    args = parser.parse_args()

    # Run main function
    main(args.data_folder, args.train_filename, args.test_filename)

