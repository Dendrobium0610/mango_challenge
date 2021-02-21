from argparse import ArgumentParser
from tqdm import tqdm
from apex import amp
from datetime import datetime
from pathlib import Path
from dataset_UDA import ImageDataSet
from torch.utils.data import DataLoader
from utils import createTrainHistory, saveDict
from sklearn.metrics import precision_score, recall_score, f1_score
from SKNet import SKNet26
import torch
import torch.nn as nn
import os
import numpy as np
import random
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
seed = 1234
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


class SupervisedClassification:
    def __init__(self, 
                 model, 
                 device, 
                 learning_rate,
                 img_dir, 
                 valid_img_dir,
                 save_dir, 
                 l2_decay,
                 batch_size, 
                 epochs, 
                 cpu_core,
                 train_txt_path, 
                 valid_txt_path, 
                 metric_item,
                 optim_name, 
                 loss, 
                 apex=True):
        
        self.device = device
        self.model = model.to(device)
        self.loss = loss.to(device)
        
        optim = getattr(torch.optim, optim_name)
        self.optim = optim(self.model.parameters(), lr=learning_rate, weight_decay=l2_decay)
        self.apex = apex
        
        if self.apex:
            self.model, self.optim = amp.initialize(self.model, self.optim, opt_level="O1")
            
        self.save_dir = save_dir
        if not os.path.isdir(str(self.save_dir)):
            os.makedirs(str(self.save_dir))

        self.epochs = epochs
        self.train_dataset = ImageDataSet(train_txt_path, 
                                          img_dir, 
                                          label_dir)
        
        self.valid_dataset = ImageDataSet(valid_txt_path, 
                                          valid_img_dir, 
                                          valid_label_dir, 
                                          sort=True)
        
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size, 
                                       shuffle=True,
                                       num_workers=cpu_core, 
                                       pin_memory=True, 
                                       drop_last=True)
        
        self.valid_loader = DataLoader(self.valid_dataset, 
                                       batch_size=batch_size, 
                                       shuffle=False,
                                       num_workers=cpu_core, 
                                       pin_memory=True, 
                                       drop_last=True)
        
        self.train_history = createTrainHistory(metric_item)
        self.best_target = 0.0
        self.best_target_epoch = 0

    def train(self):
        self.model.train()
        t_loss = 0.0
        with tqdm(total=len(self.train_dataset), desc="train ", unit="img",
                  bar_format='{l_bar}{bar:160}{r_bar}{bar:-10b}') as pbar:
            
            for imgs, labels in self.train_loader:
                imgs, labels = imgs.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.long)

                pred = self.model(imgs)
                loss = self.loss(pred, labels)
                
                self.optim.zero_grad()
                if self.apex:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optim.step()

                t_loss += loss.item()
                pbar.update(imgs.shape[0])
                
        self.train_history["train"]["loss"].append(t_loss / len(self.train_loader))

        
    def valid(self):
        self.model.eval()
        v_loss = 0.0

        with torch.no_grad():
            with tqdm(total=len(self.valid_dataset), desc="valid ", unit="img",
                      bar_format='{l_bar}{bar:160}{r_bar}{bar:-10b}') as pbar:
                target = []
                prediction = []
                for imgs, labels in self.valid_loader:
                    imgs, labels = imgs.to(self.device, dtype=torch.float32), labels.to(self.device, dtype=torch.long)

                    pred = self.model(imgs)
                    loss = self.loss(pred, labels)
                    pred = torch.sigmoid(pred).ge(0.5)
                    prediction.extend(pred.cpu().numpy())
                    target.extend(labels.cpu().numpy())
                    v_loss += loss.item()              

                    pbar.update(imgs.shape[0])
                    
        v_SE = recall_score(y_true=target, y_pred=prediction, average="macro")
        v_PR = precision_score(y_true=target, y_pred=prediction, average="macro")
        v_F1 = f1_score(y_true=target, y_pred=prediction, average="macro")
                    
        self.train_history["valid"]["loss"].append(v_loss / len(self.valid_loader))
        self.train_history["valid"]["SE"].append(v_SE)
        self.train_history["valid"]["PR"].append(v_PR)
        self.train_history["valid"]["F1"].append(v_F1)

    def start(self):
        begin_time = datetime.now()
        print("*" * 150)
        print("training start at ", begin_time)
        print("Torch Version : ", torch.__version__)
        print("Device : ", self.device)
        print("*" * 150)
        for epoch in range(self.epochs):
            print("*" * 150)
            print("Epoch %d/%d \n" % (epoch + 1, self.epochs))
            start_time = datetime.now()
            self.train()
            self.valid()
            
            end_time = datetime.now()
            epoch_total_time = end_time - start_time
            
            self.show_epoch_results(epoch, epoch_total_time)
            self.save_best_target(epoch)
            saveDict("%s/train_history.pickle" % (str(self.save_dir)), self.train_history)
            print("Epoch %d end" % (epoch + 1))
            
        finish_time = datetime.now()
        print("*" * 150)
        print("training end at ", finish_time)
        print("Total Training Time : ", finish_time - begin_time)
        print("*" * 150)

    def save_best_target(self, epoch):
        if (self.train_history["valid"]["F1"][epoch]) > self.best_target:
            self.best_target_epoch = epoch + 1
            last_best = self.best_target
            self.best_target = self.train_history["valid"]["F1"][epoch]
            improve_result = f"F1 improves from {last_best:2.5f} to {self.best_target:2.5f}"
            print(improve_result)
            torch.save(self.model.state_dict(), self.save_dir / "bestF1.pth")
            print("save model to %s" % (str(self.save_dir / "bestF1.pth")))
        else:
            improve_result = f"valid_F1 did not improve from {self.best_target:2.5f} " \
                             f"since Epoch {self.best_target_epoch:d}"
            print(improve_result)

    def show_epoch_results(self, epoch, epoch_total_time):

        result = f"Epoch {epoch + 1} time : {epoch_total_time.seconds} secs, " \
                 f"loss : {self.train_history['train']['loss'][epoch]:2.5f}, " \
                 f"valid_loss : {self.train_history['valid']['loss'][epoch]:2.5f}, " \
                 f"valid_PR : {self.train_history['valid']['PR'][epoch]:2.5f}, " \
                 f"valid_SE : {self.train_history['valid']['SE'][epoch]: 2.5f}, " \
                 f"valid_F1 : {self.train_history['valid']['F1'][epoch]:2.5f}"
        print(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--b", "--batch_size", default=64, type=int)
    parser.add_argument("--e", "--Epoch", default=100, type=int)
    parser.add_argument("--c", "--cpu_core", default=8, type=int)
    parser.add_argument("--d", "--device",
                        default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--loss", "--su_loss", default=nn.BCEWithLogitsLoss())
    parser.add_argument("--s_dir", "--save_dir", default=Path("./mango/record/SKNet26"), type=pathlib.PosixPath)
    parser.add_argument("--t_i_dir", "--train_img_dir", default=Path("./mango/Train"), type=pathlib.PosixPath)
    parser.add_argument("--v_i_dir", "--valid_img_dir", default=Path("./mango/Dev"), type=pathlib.PosixPath) 
    parser.add_argument("--train_csv_path", default="./mango/train_final.csv", type=str)
    parser.add_argument("--valid_csv_path", default="./mango/dev_final.csv", type=str)
    parser.add_argument("--m1", "--model", default=SKNet26(nums_class=5))
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--l2", "--weight_decay", default=1e-4, type=float)
    parser.add_argument("--k", "--metric", default=["loss", "SE", "SP", "PR", "F1"], type=list)
    args = parser.parse_args()

    train = SupervisedClassification(model=args.m, 
                                     batch_size=args.b, 
                                     loss=args.loss,
                                     device=args.d, 
                                     learning_rate=args.lr, 
                                     l2_decay=args.l2, 
                                     img_dir=args.i_dir,
                                     valid_img_dir=args.v_i_dir,
                                     save_dir=args.s_dir, 
                                     cpu_core=args.c, 
                                     metric_item=args.k,
                                     epochs=args.e, 
                                     train_csv_path=args.train_csv_path, 
                                     valid_csv_path=args.valid_csv_path,
                                     optim_name="Adam")
    train.start()
