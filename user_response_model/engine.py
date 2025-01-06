import torch
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, log_loss, precision_recall_fscore_support
import numpy as np

def use_optimizer(network, params):
    if params == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
#                                      lr=params['lr'],
#                                      weight_decay=params['l2_regularization'])
                                     lr=0.01,
                                     weight_decay=5e-6)
    elif params == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer
        

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, model):
        self.model = model.cuda()  # model configuration
        self.opt = use_optimizer(self.model, 'adam')
        # implicit feedback
        self.crit = torch.nn.BCELoss()

    def train_single_batch(self, user, seq_data, pos_data, pos_label):
        indices = np.where(np.array(pos_data) != 0)
        user = torch.LongTensor(np.array(user)).cuda()
        seq_data = torch.LongTensor(np.array(seq_data)).cuda()
        pos_data = torch.LongTensor(np.array(pos_data)).cuda()
        pos_label = torch.FloatTensor(np.array(pos_label)).cuda()
        self.opt.zero_grad()     
        pos_logits = self.model(user, seq_data, pos_data)
        loss = self.crit(pos_logits[indices], pos_label[indices])
        loss.backward()
        self.opt.step()

    def train_an_epoch(self, sampler, num_batch):
        self.model.train()
        for step in tqdm(range(num_batch)):
            user, seq_data, pos_data, pos_label = sampler.next_batch()
            self.train_single_batch(user, seq_data, pos_data, pos_label)
        

    def save_para(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        
    def load_para(self, save_path):
        self.model.load_state_dict(torch.load(save_path))
        
    def test(self, processed_user_test_data, processed_user_test_label):
        user_indices, log_seqs, item_indices = processed_user_test_data
        batch_idx_list = list(range(len(user_indices)))
#         batch_data = [batch_idx_list[i: i+self.config['eval_batch_size']] for i in range(0, len(batch_idx_list), self.config['eval_batch_size'])]
        batch_data = [batch_idx_list[i: i+32] for i in range(0, len(batch_idx_list), 32)]
        self.model.eval()
        with torch.no_grad():
            temp = 0
            for batch_idx in batch_data:
                batch_user_indices = user_indices[batch_idx]
                batch_log_seqs = log_seqs[batch_idx]
                batch_item_indices = item_indices[batch_idx]
                batch_pred = self.model.predict(batch_user_indices, batch_log_seqs, batch_item_indices).squeeze()
                if temp == 0:
                    preds = batch_pred
                else:
                    preds = torch.cat((preds, batch_pred))
                temp += 1
            labels = processed_user_test_label
                    
        auc = roc_auc_score(np.array(labels), preds.cpu().numpy())
        logloss = log_loss(np.array(labels), preds.cpu().numpy().astype("float64"))

        return auc, logloss, None, None, None

    def infer(self, processed_user_test_data, processed_user_test_label):
        user_indices, log_seqs, item_indices = processed_user_test_data
        batch_idx_list = list(range(len(user_indices)))
        batch_data = [batch_idx_list[i: i+32] for i in range(0, len(batch_idx_list), 32)]
        self.model.eval()
        with torch.no_grad():
            temp = 0
            for batch_idx in batch_data:
                batch_user_indices = user_indices[batch_idx]
                batch_log_seqs = log_seqs[batch_idx]
                batch_item_indices = item_indices[batch_idx]
                batch_pred = self.model.predict(batch_user_indices, batch_log_seqs, batch_item_indices)
                if temp == 0:
                    preds = batch_pred
                else:
                    preds = torch.cat((preds, batch_pred))
                temp += 1
            labels = processed_user_test_label
        return preds
    