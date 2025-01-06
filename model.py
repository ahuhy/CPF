
import torch
from torch import nn
import pandas as pd
import torch.nn.init as init
from math import sqrt
import torch.nn.functional as F
from scipy.stats import pointbiserialr
import warnings


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CPF(nn.Module):
    def __init__(self, n_at, n_it, n_exercise, n_question, d_a, d_e, d_k, d_l, d_f, q_matrix, p_matrix, dropout=0.2):
        super(CPF, self).__init__()
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.d_l = d_l
        self.d_f = d_k
        self.q_matrix = q_matrix
        self.p_matrix = p_matrix
        self.n_question = n_question       

        self.at_embed = nn.Embedding(n_at + 10, d_k)
        torch.nn.init.xavier_uniform_(self.at_embed.weight)
        self.it_embed = nn.Embedding(n_it + 10, d_k)
        torch.nn.init.xavier_uniform_(self.it_embed.weight)
        self.e_embed = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
        self.k_embed = nn.Embedding(n_question + 10, d_k)  
        torch.nn.init.xavier_uniform_(self.k_embed.weight)
        self.d_embed = nn.Embedding(n_exercise + 10, d_k)  
        torch.nn.init.xavier_uniform_(self.d_embed.weight)
        self.al_embed = nn.Embedding(d_l + 10, d_k)  
        torch.nn.init.xavier_uniform_(self.al_embed.weight)
        self.e_discrimination = nn.Embedding(n_exercise + 10, d_k)
        torch.nn.init.xavier_uniform_(self.e_embed.weight)
       

        self.linear_1 = nn.Linear(d_a + d_e + d_k * 2   , d_k)
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_2 = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_3 = nn.Linear(2 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_3.weight)
        self.linear_4 = nn.Linear(4 * d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_4.weight)
        self.linear_5 = nn.Linear(d_e + d_k * 3 , d_k)
        torch.nn.init.xavier_uniform_(self.linear_5.weight)
        self.linear_6 = nn.Linear(d_k * 3 + d_e , d_k)
        torch.nn.init.xavier_uniform_(self.linear_6.weight)
        self.linear_7 = nn.Linear(d_k + d_k, d_k)
        torch.nn.init.xavier_uniform_(self.linear_7.weight)
        self.linear_8 = nn.Linear(d_k * 5 , d_k)
        torch.nn.init.xavier_uniform_(self.linear_8.weight)

     

        self.linear_a = nn.Linear(d_k * 2  , d_k)
        torch.nn.init.xavier_uniform_(self.linear_a.weight)
        self.linear_f = nn.Linear(d_k * 4  , d_k)
        torch.nn.init.xavier_uniform_(self.linear_f.weight)
        self.linear_b = nn.Linear(d_k * 2  , d_k)
        torch.nn.init.xavier_uniform_(self.linear_b.weight)
        self.linear_c = nn.Linear(d_k * 2  , d_k)
        torch.nn.init.xavier_uniform_(self.linear_c.weight)

        self.update_matrixq= nn.Parameter(torch.ones(n_exercise + 1, n_question + 1))
        self.update_matrixp= nn.Parameter(torch.ones(n_question + 1,n_question + 1))  
        self.attention = nn.Linear(d_k, 1)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.dropout = nn.Dropout(dropout)
        self.prednet_full1 = nn.Linear(d_k, d_k)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(d_k, d_k)
        self.drop_2 = nn.Dropout(p=0.5)



    def forward(self, k_data, e_data, at_data, a_data, it_data,al_data, df_data):
        batch_size, seq_len = e_data.size(0), e_data.size(1)
        e_embed_data = self.e_embed(e_data)
        at_embed_data = self.at_embed(at_data)
        it_embed_data = self.it_embed(it_data)
        k_embed_data = self.k_embed(k_data)
        df_embed_data = self.d_embed(df_data)
        al_embed_data = self.al_embed(al_data)
        aa_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        h_pre = nn.init.xavier_uniform_(torch.zeros(self.n_question + 1, self.d_k)).repeat(batch_size, 1, 1).to(device)
        h_tilde_pre = None
        
        a = nn.Parameter(torch.tensor([0.09], requires_grad=True)).to(device)
        b = nn.Parameter(torch.tensor([0.9], requires_grad=True)).to(device)
        r = nn.Parameter(torch.tensor([0.01], requires_grad=True)).to(device)
        student_ability = a * df_embed_data + b * al_embed_data + r * at_embed_data


        e_discrimination = torch.sigmoid(self.e_discrimination(e_data))# * 10
        e_discrimination_data = e_discrimination * (student_ability - df_embed_data) #* kn_emb

        all_learning = self.linear_1(torch.cat((e_embed_data, k_embed_data, aa_data, student_ability), 2))
        learning_pre = torch.zeros(batch_size, self.d_k).to(device)
        pred = torch.zeros(batch_size, seq_len).to(device)


        for t in range(0, seq_len - 1):           
            e = e_data[:, t]
            # q_e: (bs, 1, n_skill)
            q_matrix = self.q_matrix * self.update_matrixq
            q_e = q_matrix[e].view(batch_size, 1, -1)
            it = it_embed_data[:, t]
            k = k_data[:, t]
            p_matrix = self.p_matrix * self.update_matrixp
            p_e = p_matrix[k]  
            lt = it_data[:, t]
            at = at_data[:, t]
         
            # Learning Module
            if h_tilde_pre is None:
                h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k)

            direct_knowledge_state = p_e.unsqueeze(1).bmm(h_pre).view(batch_size, self.d_k)
            direct_knowledge_state = self.tanh(direct_knowledge_state)

            learning = all_learning[:, t]
            sa = student_ability[:, t]
            # df = df_embed_data[:, t]
            # al = al_embed_data[:, t]
            # at = at_embed_data[:, t]
         
            learning_gain = self.linear_2(torch.cat((learning, h_tilde_pre), 1))
            learning_gain = self.tanh(learning_gain)

          
            gamma_l = self.linear_3(torch.cat((learning, h_tilde_pre), 1))
            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))  + F.normalize(self.sig(direct_knowledge_state), p=2, dim=-1).unsqueeze(1).repeat(1, 103, 1)

       
            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)

            forget_w = nn.Parameter(torch.tensor([1.0], requires_grad=True)).to(device)
            delta_t = torch.abs((it_data[:, t] + at_data[:, t]) - (it_data[:, t+1] + at_data[:, t+1]))

            close = 10
            topk_delta_t, topk_indices = torch.topk(delta_t + 1e-6, k=close, largest=False)
            min_delta_t = torch.mean(topk_delta_t, dim=0)
            nearest_indices = topk_indices
            nearest_prerequisites = p_e[torch.arange(batch_size).unsqueeze(1), nearest_indices]
            tau = nn.Parameter(torch.tensor([0.3], requires_grad=True)).to(device)
            gamma = nn.Parameter(torch.tensor([1.0], requires_grad=True)).to(device)
            new_forget_w = self.sig(1 / (1 + torch.exp((delta_t.unsqueeze(1) - min_delta_t.unsqueeze(0) + tau) / gamma)))  
            zero_prerequisite_mask = (nearest_prerequisites == 0).any(dim=1)
            new_forget_w[zero_prerequisite_mask] = 1.0
            forget_w.data = new_forget_w
        

            n_skill = LG_tilde.size(1)
            LG_t = LG * forget_w.repeat(1, self.d_k)
            
            # Forgetting Module
            # h_pre: (bs, n_skill, d_k)
            # LG: (bs, d_k)
            # it: (bs, d_k)

            gamma_f = self.sig(self.linear_4(torch.cat((
                h_pre,
                LG_t.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                it.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                sa.repeat(1, n_skill).view(batch_size, -1, self.d_k)
            ), 2)))           

            h = LG_tilde + h_pre * gamma_f


            # Predicting Module 

            h_tilde = self.q_matrix[e_data[:, t + 1]].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k)
            y = self.sig(self.linear_6(torch.cat((e_embed_data[:, t + 1], k_embed_data[:, t+1], e_discrimination_data[:,t+1] , h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y

            
            # prepare for next prediction
            learning_pre = learning
            h_pre = h 
            h_tilde_pre = h_tilde


        return pred
