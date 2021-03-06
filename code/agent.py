# coding: utf-8

import torch
import numpy as np
from math import ceil
from torch import nn
from ipdb import set_trace
import time
import random

class lstm(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm, self).__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.first_flag = True
    
    def forward(self, x):
        if self.first_flag:
            self.h, self.c = self.cell(x)
        else:
            self.h, self.c = self.cell(x, (self.h, self.c))
        return self.h



class Agent(nn.Module):
    ''' Agent for collecting evidence. '''

    def __init__(self, env, embeddings, edge_embeds, episode_len=10, \
                    embed_dim=64, lstm_hidden=64, dropout=0.5):
        super(Agent, self).__init__()
        self.env = env
        self.embeddings = embeddings.cuda()    # torch.nn.Embedding
        self.edge_embeds = edge_embeds.cuda()
        self.episode_len = episode_len
        self.embed_dim = embed_dim

        self.lstm_cell = lstm(input_size=4 * embed_dim, hidden_size=lstm_hidden)

        self.fc_layer = nn.Sequential(
            nn.Linear(lstm_hidden, out_features=lstm_hidden),
            # nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(lstm_hidden, embed_dim),
            # nn.Dropout(dropout),
            # nn.ReLU(),
        )

    def forward(self, start_inds, end_inds):
        ''' Generate an episode.
        start_inds/ end_inds: torch.long tensor with size [1]
        '''
        start_embeds = self.embeddings(start_inds)
        end_embeds = self.embeddings(end_inds)
        # return torch.zeros([start_inds.size(0), 1]), torch.zeros([start_inds.size(0), 1]), start_embeds, end_embeds

        query = torch.cat([start_embeds, end_embeds], dim=-1)

        # first epoch, 'a' is initialized as zeros
        B = start_inds.size(0)
        a = torch.cat([torch.zeros([B, self.embed_dim]).cuda(), start_embeds], dim=-1)

        current_inds, current_embeds, prev_edge_embeds = self.sub_forward(a, query, start_inds)

        output_embeds = [start_embeds, current_embeds]
        output_inds = [start_inds, current_inds]

        for _ in range(self.episode_len - 2):
            a = torch.cat([prev_edge_embeds, current_embeds], dim=-1)
            current_inds, current_embeds, prev_edge_embeds = self.sub_forward(a, query, current_inds)

            output_embeds.append(current_embeds)
            output_inds.append(current_inds)
        lists = np.array([ts.tolist() for ts in output_inds + [end_inds]])
        return output_embeds, output_inds, start_embeds, end_embeds


    def sub_forward(self, a, query, current_inds):
        ''' Perform a sub forward step.
        a:              previous action and current node embeddings
        query:          query embeddings
        current_inds:   index indicating the current node.
        '''
        input_ = torch.cat([a, query], dim=-1)  # [B,], [B, nd]
        x = self.lstm_cell(input_)

        # mlp_policy
        x = self.fc_layer(x)
        # begin = time.time()
        # find next actions (indices), and extract corresponding embeddings
        node_inds, edge_types = self.env.get_action_new(current_inds) # 输入为[B, 1]的index，记得加入一个no op action (batched)
        # print('sub_forward', (time.time() - begin) * 100)

        cand_edge_embeds = self.edge_embeds(edge_types)  # [B, n, hidden]
        candidates = self.embeddings(node_inds)    # [B, n, hidden]

        # compute probability and conduct sampling
        d = torch.nn.functional.softmax(torch.einsum('anh, ah->an', candidates, x), dim=-1)
        tmp_inds = d.multinomial(1).view(-1)    # sample
        row_inds = np.arange(tmp_inds.size(0))
        # get next node indices and embeddings, return
        next_inds = node_inds[row_inds, tmp_inds]
        next_embeds = candidates[row_inds, tmp_inds]
        edge_embeds = cand_edge_embeds[row_inds, tmp_inds]

        return next_inds, next_embeds, edge_embeds



class MultiAgents(nn.Module):
    ''' Aggregated multiple agents. '''
    def __init__(self, env, embeddings, edge_embeds, n_agents=4, \
                    episode_len=10, embed_dim=64, lstm_hidden=64, dropout=0.5):
        super(MultiAgents, self).__init__()
        self.agents = nn.ModuleList()
        for _ in range(n_agents):
            self.agents.append(Agent(env, embeddings, edge_embeds, \
                        episode_len, embed_dim, lstm_hidden, dropout))


    def __call__(self, batch_start_inds, batch_end_inds):
        ''' Generate an episode for multiple agents 
        batch_start/end_inds should be integer Tensors of shape [B, n_start/end]
        '''
        # 500, 4
        B, n = batch_start_inds.size()
        outputs_embs, outputs_ids = [], []

        batch_start_inds = batch_start_inds.reshape(-1)
        batch_end_inds = batch_end_inds.reshape(-1)


        for ag in self.agents:
            output_embeds, output_inds, start_embeds, end_embeds = ag(batch_start_inds, batch_end_inds)
            stacked_embeds = torch.stack(output_embeds).transpose(0, 1)
            stacked_inds = torch.stack(output_inds).transpose(0, 1)

            stacked_embeds = stacked_embeds.view(B, n, stacked_embeds.size(1), stacked_embeds.size(2))
            stacked_inds = stacked_inds.view(B, n, stacked_inds.size(1))
            # print(stacked_embeds.size(), stacked_inds.size())
            # exit()
            outputs_embs.append(stacked_embeds)
            outputs_ids.append(stacked_inds)
        
        
        outputs_embs = torch.stack(outputs_embs).transpose(0, 1)
        outputs_ids = torch.stack(outputs_ids).transpose(0, 1)
        # print(outputs_embs.size()) # [500, 3, 4, 2, 64]
        # exit()
        
        # for b in range(B):
        #     batchoutput_embs, batchoutput_ids = [], []
        #     for ag in self.agents:
        #         output_embeds, output_inds = ag(batch_start_inds[b], batch_end_inds[b])
        #         # output_embeds 10个3 * 64
        #         # output_inds
        #         stacked_embeds = torch.stack(output_embeds).transpose(0, 1)
        #         stacked_ids = torch.stack(output_inds).transpose(0, 1)
        #         batchoutput_embs.append(stacked_embeds)
        #         batchoutput_ids.append(stacked_ids)
        #     outputs_embs.append(torch.stack(batchoutput_embs))
        #     outputs_ids.append(torch.stack(batchoutput_ids))
        # print(torch.stack(output_embeds).size())
        # exit()
        # outputs_embs = torch.stack(outputs_embs), 
        # outputs_ids = torch.stack(outputs_ids)
        return outputs_embs, outputs_ids, start_embeds, end_embeds



# ' 单Agent测试 '
# # 初始化agent
# agent = Agent(env, embeddings, edge_embeds, 5)

# # 尝试生成起始点为0，终止点为1的路径
# start_inds = torch.tensor([0,1,2], dtype=torch.long)
# end_inds = torch.tensor([1,2,3], dtype=torch.long)

# output_embeds, output_inds = agent(start_inds, end_inds)
# print(len(output_embeds), [i.size() for i in output_embeds])
# print(output_inds)
# output_embeds, output_inds = agent(start_inds, end_inds)
# print(output_inds)

if __name__ == '__main__':
    embeddings = nn.Embedding()
    agents = Agent(env, embeddings, edge_embeds, 5)
