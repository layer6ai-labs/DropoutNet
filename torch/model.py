import torch
import torch.nn as nn
import scipy
import numpy as np

from tqdm import tqdm


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

@torch.no_grad()
def init_weights(net):
    if type(net) == nn.Linear:
        #torch.nn.init.normal_(net.weight, mean=0, std=0.01)
        truncated_normal_(net.weight, std=0.01)
        if net.bias is not None:
            torch.nn.init.constant_(net.bias, 0)


def get_model(latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):
    model = DeepCF(latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out)
    model.apply(init_weights)
    return model
        


class TanHBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(TanHBlock, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.bn = nn.BatchNorm1d(
                num_features=dim_out,
                momentum=0.01,
                eps=0.001
                )

    
    def forward(self, x):
        out = self.layer(x)
        out = self.bn(out)
        out = torch.tanh(out)
        return out

class DeepCF(nn.Module):
    """
    main model class implementing DeepCF
    also stores states for fast candidate generation
    latent_rank_in: rank of preference model input
    user_content_rank: rank of user content input
    item_content_rank: rank of item content input
    model_select: array of number of hidden unit,
        i.e. [200,100] indicate two hidden layer with 200 units followed by 100 units
    rank_out: rank of latent model output
    """

    def __init__(self, latent_rank_in, user_content_rank, item_content_rank, model_select, rank_out):
        super(DeepCF, self).__init__()
        self.rank_in = latent_rank_in
        self.phi_u_dim = user_content_rank
        self.phi_v_dim = item_content_rank
        self.model_select = model_select
        self.rank_out = rank_out

        # inputs
        self.phase = None
        self.target = None
        self.eval_trainR = None
        self.U_pref_tf = None
        self.V_pref_tf = None
        self.rand_target_ui = None

        # outputs in the model
        self.updates = None

        # predictor
        self.tf_topk_vals = None
        self.tf_topk_inds = None
        self.preds_random = None
        self.tf_latent_topk_cold = None
        self.tf_latent_topk_warm = None
        self.eval_preds_warm = None
        self.eval_preds_cold = None
        
        u_dim = self.rank_in + self.phi_u_dim if self.phi_u_dim > 0 else self.rank_in
        v_dim = self.rank_in + self.phi_v_dim if self.phi_v_dim > 0 else self.rank_in

        print ('\tu_concat rank=%s' % str(u_dim))
        print ('\tv_concat rank=%s' % str(v_dim))
        
        u_dims = [u_dim] + self.model_select
        v_dims = [v_dim] + self.model_select
        self.u_layers = nn.ModuleList(TanHBlock(u_dims[i], u_dims[i + 1]) for i in range(len(u_dims) - 1))
        self.v_layers = nn.ModuleList(TanHBlock(v_dims[i], v_dims[i + 1]) for i in range(len(v_dims) - 1))
        
        self.u_emb = nn.Linear(u_dims[-1], self.rank_out)
        self.v_emb = nn.Linear(v_dims[-1], self.rank_out)

    def encode(self, Uin, Vin, Ucontent, Vcontent):
        
        if self.phi_u_dim>0:
            u_concat = torch.cat((Uin, Ucontent), 1)
        else:
            u_concat = Uin

        if self.phi_v_dim>0:
            v_concat = torch.cat((Vin, Vcontent), 1)
        else:
            v_concat = Vin
            
        u_out = u_concat
        for layer in self.u_layers:
            u_out = layer(u_out)
        U_embedding = self.u_emb(u_out)
        
        v_out = v_concat
        for layer in self.v_layers:
            v_out = layer(v_out)
        V_embedding = self.v_emb(v_out)
        return U_embedding, V_embedding
        
    def forward(self, Uin, Vin, Ucontent, Vcontent):
        
        U_embedding, V_embedding = self.encode(Uin, Vin, Ucontent, Vcontent)
        
        preds = U_embedding * V_embedding
        preds = torch.sum(preds, 1)
        return preds, U_embedding, V_embedding

    @torch.no_grad()
    def evaluate(self, recall_k, eval_data, device=None):
        """
        given EvalData runs batch evaluation
        :param recall_k: list of thresholds to compute recall at (information retrieval recall)
        :param eval_data: EvalData instance
        :return: recall array at thresholds matching recall_k
        """
        d = device

        tf_eval_preds_batch = []
        for (batch, (eval_start, eval_stop)) in enumerate(tqdm(eval_data.eval_batch, desc='eval', leave=False)):

            Uin = eval_data.U_pref_test[eval_start:eval_stop, :]
            Vin = eval_data.V_pref_test
            Vcontent = eval_data.V_content_test

            if self.phi_u_dim > 0: 
                Ucontent= eval_data.U_content_test[eval_start:eval_stop, :]
            else:
                Ucontent = None

            Uin = torch.tensor(Uin)
            Vin = torch.tensor(Vin)
            if Ucontent is not None:
                Ucontent = torch.tensor(Ucontent)
            if Vcontent is not None:
                Vcontent = torch.tensor(Vcontent)
            if d is not None:
                Uin = Uin.to(d)
                Vin = Vin.to(d)
                Ucontent = Ucontent.to(d)
                Vcontent = Vcontent.to(d)
            U_embedding, V_embedding = self.encode(Uin, Vin, Ucontent, Vcontent)
            embedding_prod = torch.matmul(U_embedding, V_embedding.t())


            if not eval_data.is_cold:
                eval_trainR = eval_data.tf_eval_train[batch]
                embedding_prod = embedding_prod + eval_trainR

            _, eval_preds = torch.topk(embedding_prod, k=recall_k[-1], sorted=True)
            tf_eval_preds_batch.append(eval_preds.detach().cpu().numpy())


        tf_eval_preds = np.concatenate(tf_eval_preds_batch)

        # filter non-zero targets
        y_nz = [len(x) > 0 for x in eval_data.R_test_inf.rows]
        y_nz = np.arange(len(eval_data.R_test_inf.rows))[y_nz]

        preds_all = tf_eval_preds[y_nz, :]

        recall = []
        for at_k in tqdm(recall_k, desc='recall', leave=False):
            preds_k = preds_all[:, :at_k]
            y = eval_data.R_test_inf[y_nz, :]

            x = scipy.sparse.lil_matrix(y.shape)
            x.data = np.array([z.tolist() for z in np.ones_like(preds_k)]+[[]],dtype=object)[:-1]
            x.rows = np.array([z.tolist() for z in preds_k]+[[]],dtype=object)[:-1]
            z = y.multiply(x)
            recall.append(np.mean(np.divide((np.sum(z, 1)), np.sum(y, 1))))
        return recall
