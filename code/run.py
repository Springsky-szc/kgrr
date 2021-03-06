from tqdm import tqdm
import json
from environment import environment
from agent import MultiAgents
import torch
import numpy as np
from torch import nn
from judge import Judge
from dataset import DoubanDataset, get_dataloader
import random
from sklearn.metrics import roc_auc_score
import time
from ipdb import set_trace
import argparse


class Conf:
    def __init__(self):
        self.dataset = "douban"
        if self.dataset == "book":
            self.graph_num = 209080
            self.edge_type_num = 40
        if self.dataset == "douban":
            self.graph_num = 196444
            self.edge_type_num = 13
        self.dim = 64
        self.batch_size = 5000

        self.LR = 1e-5
        self.EPOCH = 2000
        self.path_num = 4
        self.n_agents = 3
        self.agent_weight = 10
        self.episode_len = 3
        # batch_size * agent_num * path_num * path_len * embedding_dim

class Model(nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()
        with open('../data/' + conf.dataset + '/graph_map.json', 'r') as f:
            graph_map = json.loads(f.read())
        self.env = environment(graph_map, '../data/' + conf.dataset + '/node2id.json')

        # 初始化embedding
        # torch.nn.Embedding(2, 2, max_norm=0.1)(torch.tensor([0, 1]))
        self.embeddings = torch.nn.Embedding(conf.graph_num, conf.dim, max_norm=0.1)#len(self.env.new_map), 64)
        self.edge_embeds = torch.nn.Embedding(conf.edge_type_num, conf.dim, max_norm=0.1)
        self.multiagents = MultiAgents(self.env, self.embeddings, self.edge_embeds, n_agents=conf.n_agents, episode_len=conf.episode_len)
        self.judge = Judge(conf)

    def forward(self, batch_start_inds, batch_end_inds, batch_label):
        # print(batch_start_inds.size())    [500, 4]
        outputs_embs, outputs_ids, start_embeds, end_embeds = self.multiagents(batch_start_inds, batch_end_inds)
        logit, loss = self.judge(outputs_embs, batch_label, start_embeds, end_embeds)
        return logit, loss


class Optimize():
    def __init__(self, model, lr):
        self.model = model
        self.judge_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def judge_opt(self, loss):
        self.judge_optimizer.zero_grad()
        loss.backward()
        self.judge_optimizer.step()

def train_step(batch_start_inds, batch_end_inds, batch_label, model, opt):
    ''' One train step
    '''
    _, loss = model(batch_start_inds, batch_end_inds, batch_label)
    opt.judge_opt(loss)
    return loss

def train(conf, loader, model, opt):
    train_loss = 0
    loader = tqdm(loader)
    for batch in loader:
        if random.randint(1, 4000) == 6666:
            a = time.time()
            model.env.create_new_map_emb()
            print('create_new_map_emb', time.time() - a)
        # print(batch[0].size())    [500, 2]
        batch_start_inds = batch[0][:, 0].repeat(conf.path_num, 1).permute(1, 0)
        # print(batch_start_inds.size()) [500, 4]
        batch_end_inds = batch[0][:, 1].repeat(conf.path_num, 1).permute(1, 0)
        batch_label = batch[1]
        # batch_start_inds = torch.tensor([[0, 0, 0], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 14]], dtype=torch.long)
        # batch_end_inds = torch.tensor([[9, 10, 11], [12, 1, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23]],
        #                               dtype=torch.long)

        loss = train_step(batch_start_inds, batch_end_inds, batch_label, model, opt)
        train_loss += float(loss.cpu())
    return train_loss

def test(conf, loader, model):
    loader = tqdm(loader)
    rst = []
    logit_list = []
    for batch in loader:
        batch_start_inds = batch[0][:, 0].repeat(conf.path_num, 1).permute(1, 0)
        batch_end_inds = batch[0][:, 1].repeat(conf.path_num, 1).permute(1, 0)
        batch_label = batch[1]
        logit, _ = model(batch_start_inds, batch_end_inds, batch_label)
        for i in range(logit.size()[0] // 10):
            rank = 1
            i_logit = logit[(i * 10): ((i + 1) * 10)][:, 0]
            for i in i_logit[1:]:
                if i <= i_logit[0]:
                    rank += 1
            rst.append(rank)
            i_logit = i_logit.cpu().detach().tolist()
            logit_list += i_logit

    name_ran = random.randint(1, 999)
    np.save('./rst/rst' + str(name_ran) + '.npy', np.array(rst))
    yz = np.mean(logit_list)
    acc = np.mean(np.equal(([1] + [0] * 9) * int(len(logit_list) / 10), [1 if i < yz else 0 for i in logit_list]))
    auc = roc_auc_score(y_true=([0] + [1] * 9) * int(len(logit_list) / 10), y_score=logit_list)
    print(auc, acc, np.mean([1 / i for i in rst]), np.mean([1 if i < 2 else 0 for i in rst]), np.mean([1 if i < 4 else 0 for i in rst]), np.mean([1 if i < 6 else 0 for i in rst]))
    return auc, acc, np.mean([1 / i for i in rst]), np.mean([1 if i < 2 else 0 for i in rst]), np.mean([1 if i < 4 else 0 for i in rst]), np.mean([1 if i < 6 else 0 for i in rst])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='douban', help='Store model path.')
    args = parser.parse_args()
    conf = Conf()
    conf.dataset = args.dataset
    # 构建数据集
    data_train_fn = '../data/' + conf.dataset + '/Interact_tuple_train.dat'
    data_test_fn = '../data/' + conf.dataset + '/Interact_tuple_test.dat'
    node2id_fn = '../data/' + conf.dataset + '/node2id.json'
    train_set = DoubanDataset(data_train_fn, node2id_fn, 'train')
    test_set = DoubanDataset(data_test_fn, node2id_fn, 'test')
    # print(dataset[0])
    train_loader = get_dataloader(train_set, conf.batch_size, True)
    test_loader = get_dataloader(test_set, conf.batch_size, False)

    # 构建模型
    model = Model(conf)
    model = model.cuda()
    opt = Optimize(model, conf.LR)


    for epoch in range(conf.EPOCH):
        if epoch % 10 == 7:
            conf.LR *= 0.9
            opt = Optimize(model, conf.LR)
        print('epoch:', epoch)

        train_loss = train(conf, train_loader, model, opt)
        test_rst = test(conf, test_loader, model)
        with open('temp' + str(conf.agent_weight) + '.txt', 'a') as f:
            rst = tuple([epoch, train_loss]) + test_rst
            f.write("epoch %d:\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % rst)
        print("epoch %d:\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n" % rst)

'''
cjy
cjyATfudan
cd kgrr/code
cat nohup.out

yqfx
yqfx
cd kgrr/code
cat nohup.out

nohup /usr/bin/python3 run.py &
'''
