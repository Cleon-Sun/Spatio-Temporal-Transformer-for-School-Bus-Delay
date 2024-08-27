from utils.setseed import set_seed
set_seed(1234)

import os

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.STTN import STTN
from utils.earlystoping import EarlyStopping
from utils.getdata import get_data
from helper import import_data_new, get_data_train_no_X, get_adj


class MySttn:
    def __init__(self, his_len=3, pre_len=1, lr=0.0001, batch_size=8, epochs=200, patience=10000):
        self.his_len = his_len
        self.pre_len = pre_len

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.lr = lr

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')

        self.modelpath = './checkpoint/STTN.pkl'

        self._get_data()
        self._get_model()

    def _get_data(self):
        file_path = "school_weekly_weather_traffic2(1).csv"
        # self.scaler = StandardScaler()
        # train, valid, test, A, x_status = get_data()
        df = import_data_new(file_path, train_test='train', weather='No',traffic='No',season_dummy='No')
        train = get_data_train_no_X(df, target_name='Avg_Delay_Time')
        self.mean = torch.mean(train).item()
        self.std = torch.std(train).item()
        train = (train - self.mean) / self.std

        df_test = import_data_new(file_path, train_test='test', weather='No',traffic='No',season_dummy='No')
        test = get_data_train_no_X(df_test, target_name='Avg_Delay_Time')
        test = (test - self.mean) / self.std

        A = get_adj("adjacency_matrixdjace.npy")
        A_wave = A + np.eye(A.shape[0])
        D_wave = np.eye(A.shape[0]) * (np.sum(A_wave, axis=0) ** (-0.5))
        self.adj = np.matmul(np.matmul(D_wave, A_wave), D_wave)
        self.adj = self.adj[:, :32]

        # train = self.scaler.fit_transform(train)
        # valid = self.scaler.transform(valid)
        # test = self.scaler.transform(test)

        # trainset = MyDataset(data=train, his_len=self.his_len, pre_len=self.pre_len)
        # validset = MyDataset(data=valid, his_len=self.his_len, pre_len=self.pre_len)
        # testset = MyDataset(data=test, his_len=self.his_len, pre_len=self.pre_len)

        trainset = MyDataset(data=train, his_len=self.his_len, pre_len=self.pre_len)
        validset = MyDataset(data=test, his_len=self.his_len, pre_len=self.pre_len)
        testset = MyDataset(data=test, his_len=self.his_len, pre_len=self.pre_len)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = STTN(adj=self.adj, len_his=self.his_len, len_pred=self.pre_len).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.95 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=True, path=self.modelpath)
        self.criterion = nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        # print(batch_x.shape)

        outputs = self.model(batch_x)
        # print(outputs.shape)
        std = torch.Tensor([self.std]).cuda()
        mean = torch.Tensor([self.mean]).cuda()
        loss = self.criterion(outputs * std + mean, batch_y * std + mean)
        return outputs, loss

    def train(self):
        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            for (batch_x, batch_y) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                valid_loss = []
                for (batch_x, batch_y) in tqdm(self.validloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y)
                    valid_loss.append(loss.item())

                test_loss = []
                for (batch_x, batch_y) in tqdm(self.testloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y)
                    test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()
            print("lr: ", self.scheduler.get_last_lr()[0])
        # self.model.load_state_dict(torch.load(self.modelpath))
        # torch.save(self.model.state_dict(), self.modelpath)

    def test(self):
        self.model.load_state_dict(torch.load(self.modelpath, map_location=self.device))
        self.model.eval()
        with torch.no_grad():
            trues, preds = [], []
            for (batch_x, batch_y) in tqdm(self.testloader):
                pred, loss = self._process_one_batch(batch_x, batch_y)
                preds.extend(pred.detach().cpu().numpy())
                trues.extend(batch_y.detach().cpu().numpy())

        trues, preds = np.array(trues), np.array(preds)

        trues = trues * self.std + self.mean
        preds = preds * self.std + self.mean

        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        print('Test: MSE:{0:.6f}, MAE:{1:.6f}'.format(mse, mae))


if __name__ == '__main__':
    my_sttn = MySttn()
    my_sttn.train()
    my_sttn.test()
