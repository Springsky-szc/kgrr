# from douban_preprocess import construct_id_dict, save_pickle, read_from_pickle
import numpy as np
import torch
import torch.nn as nn
import random
from ipdb import set_trace
from torch.autograd import Variable

# dataset, sequences = read_from_pickle('../../../cjy/kgrr/data/douban/dataset_and_sequences.dat')

class Judge(nn.Module):
    def __init__(self, conf):
        super(Judge, self).__init__()
        # self.fc = nn.Linear(conf.dim * 2, 2)
        num1, num2 = 64, 64
        # self.fc2 = nn.Sequential(nn.Linear(conf.dim, num2), nn.ReLU(), nn.Linear(num2, num2)).cuda()
        self.fc2 = nn.Sequential(nn.Linear(64 * 4, 111), nn.ReLU(), nn.Linear(111, 64)).cuda()
        self.fc3 = nn.Sequential(nn.Linear(64 * 2, 1))#, nn.ReLU(), nn.Linear(64 * 2, 64)).cuda()
        self.EPOCH = conf.EPOCH
        self.LR = conf.LR
        self.agent_weight = conf.agent_weight * 0.0
        self.batch_size = conf.batch_size
        self.dim = conf.dim
        self.att_mat = torch.Tensor(self.dim, self.dim).cuda()
        self.w = torch.Tensor(self.dim, 1).cuda()
        self.W = torch.Tensor(self.dim, self.dim).cuda()
        self.w_agent = torch.Tensor(self.dim, 1).cuda()
        self.W_agent = torch.Tensor(self.dim, self.dim).cuda()

        torch.nn.init.normal_(self.att_mat, std=0.5)
        torch.nn.init.normal_(self.w, mean=0, std=0.5)
        torch.nn.init.normal_(self.w_agent, mean=0, std=0.5)
        torch.nn.init.normal_(self.W, mean=0, std=0.5)
        torch.nn.init.normal_(self.W_agent, mean=0, std=0.5)
        torch.nn.init.normal_(self.fc3[0].weight.data, mean=1, std=0.01)
        torch.nn.init.normal_(self.fc2[0].weight.data, mean=0, std=0.5)

    def f(self, x):
        x = x.reshape((x.size()[0], x.size()[1], -1))
        x = self.fc3(x)
        return x

    def f_agent(self, x):
        x = x.reshape((x.size()[0], x.size()[1], -1))
        x = self.fc2(x)
        return x

    def forward(self, paths_emb, label, start_embeds, end_embeds):
        # return self.forward_judge(paths_emb, label, start_embeds, end_embeds)
        return self.forward_judge_and_agents1(paths_emb, label, start_embeds, end_embeds)

    def forward_judge(self, paths_emb, label, start_embeds, end_embeds):
        # torch.Size([500, 3, 4, 2, 64]) torch.Size([500]) torch.Size([2000, 64]) torch.Size([2000, 64])
        start_embeds = start_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0:1, :, :] # B, n, 1, dim
        end_embeds = end_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0:1, :, :] # B, n, 1, dim
        total_paths_emb = paths_emb.reshape((paths_emb.size(0), -1, 1, paths_emb.size()[-1])) # B, n, 1, dim
        # print(total_paths_emb.size())   # [500, 24, 1, 64]
        start_embeds = start_embeds.repeat((1, total_paths_emb.size(1), 1, 1)) # B, n, 1, dim
        end_embeds = end_embeds.repeat((1, total_paths_emb.size(1), 1, 1)) # B, n, 1, dim
        y_n = self.f(torch.cat((total_paths_emb, start_embeds, end_embeds, start_embeds * end_embeds), 2))
        # print(y_n.size()) # [500, 24, 64]

        # t_tao = torch.sigmoid(torch.matmul(torch.matmul(torch.relu(torch.sum(y_n, 1)), self.W), self.w))
        t_tao = torch.sigmoid(torch.matmul(torch.relu(torch.matmul(torch.sum(y_n, 1), self.W)), self.w))
        logit = torch.cat((t_tao, 1 - t_tao), 1)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logit, label)
        return logit, loss

    def forward_judge_and_agents1(self, paths_emb, label, start_embeds, end_embeds):
        start_embeds = start_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0, 0, :]
        end_embeds = end_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0, 0, :]
        paths_emb = torch.mean(torch.mean(torch.mean(paths_emb, 1), 1), 1)
        loss_func = nn.CrossEntropyLoss()
        logit = torch.sigmoid(torch.sum(start_embeds * (end_embeds + paths_emb), 1) * 10).reshape((-1, 1))
        loss = loss_func(torch.cat((logit, 1 - logit), 1), label)
        return logit, loss

    def forward_judge_and_agents(self, paths_emb, label, start_embeds, end_embeds):
        start_embeds = start_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0:1, :, :] # B, n, 1, dim
        end_embeds = end_embeds.reshape((paths_emb.size(0), -1, 1, self.dim))[:, 0:1, :, :] # B, n, 1, dim
        total_paths_emb = paths_emb.reshape((paths_emb.size(0), -1, 1, paths_emb.size()[-1])) # B, n, 1, dim
        # print(total_paths_emb.size())   # [500, 24, 1, 64]
        start_embeds = start_embeds.repeat((1, total_paths_emb.size(1), 1, 1)) # B, n, 1, dim
        end_embeds = end_embeds.repeat((1, total_paths_emb.size(1), 1, 1)) # B, n, 1, dim
        # y_n = self.f(torch.cat((total_paths_emb, start_embeds, end_embeds), 2))
        # print(y_n.size()) # [500, 24, 64]
        # y_n = self.f(torch.cat((total_paths_emb, start_embeds, end_embeds, start_embeds * end_embeds), 2))
        y_n_agent = self.f_agent(torch.cat((total_paths_emb, start_embeds, end_embeds, start_embeds * end_embeds), 2))
        # y_n = self.f(torch.cat((total_paths_emb, start_embeds * end_embeds), 2))
        y_n = torch.mean(torch.mean(torch.cat((total_paths_emb, start_embeds * end_embeds), 3), 1), 1)
        # set_trace()
        ''' loss for judge '''
        # t_tao = torch.sigmoid(torch.matmul(torch.matmul(torch.relu(torch.sum(y_n, 1)), self.W), self.w))
        # t_tao = torch.sigmoid(torch.mean(y_n, 1))
        t_tao = torch.sigmoid(self.fc3(y_n))
        # set_trace()
        y_n = torch.sum(torch.mean(torch.mean(start_embeds * end_embeds, 1), 1), 1)
        t_tao = torch.sigmoid(y_n.reshape((-1, 1)) * 10)
        logit = torch.cat((t_tao, 1 - t_tao), 1)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logit, label)

        ''' loss (negative rewards) for agents '''
        # t = torch.matmul(torch.relu(torch.matmul(y_n, self.W_agent)), self.w_agent).squeeze()
        t = torch.matmul(torch.relu(torch.matmul(y_n_agent, self.W)), self.w).squeeze()
        t = -1 * torch.log(torch.sigmoid(-t))
        loss_agents = torch.mean(torch.mean(t, 1))
        # print(loss, loss_agents)
        # exit()
        loss += loss_agents * self.agent_weight

        return logit, loss

    # def forward(self, paths_emb, label, start_embeds, end_embeds):
    #     start_embeds = start_embeds.reshape((paths_emb.size()[0], -1, self.dim))[:, 0, :]
    #     end = end_embeds.reshape((paths_emb.size()[0], -1, self.dim, 1))[:, 0, :]
    #     end_embeds = end_embeds.reshape((paths_emb.size()[0], -1, self.dim))[:, 0, :]
    #     total_paths_emb = paths_emb.reshape((paths_emb.size()[0], -1, paths_emb.size()[-1])) # 3000, 2, 64
    #
    #     wt = torch.unsqueeze(torch.sigmoid(total_paths_emb.matmul(self.att_mat).matmul(end)[:, :, 0]), 1) # 3000, 1, 2
    #     paths_sum = wt.matmul(total_paths_emb).squeeze(1)
    #     all_embeds = torch.cat((start_embeds, end_embeds, start_embeds * end_embeds, paths_sum), 1)
    #     # all_embeds = torch.cat((start_embeds, end_embeds, start_embeds * end_embeds, paths_sum), 1)
    #     # all_embeds = torch.cat((start_embeds, end_embeds, start_embeds * end_embeds, torch.mean(total_paths_emb, 1)), 1)
    #     logit = self.fc(all_embeds)
    #     loss_func = nn.CrossEntropyLoss()
    #     logit = torch.softmax(logit, 1)
    #     loss = loss_func(logit, label)
    #     return logit, loss
