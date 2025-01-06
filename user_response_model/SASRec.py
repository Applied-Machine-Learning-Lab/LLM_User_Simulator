import pandas as pd
import numpy as np
import torch.nn as nn
import pickle
import torch
import torch.optim as optim
from multiprocessing import Process, Queue
import logging
import datetime
from tqdm import tqdm
import os
from user_response_model.engine import Engine
# from engine import Engine
import random
        
def initLogging(logFilename):
    """Init for logging
    """
    logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    
# train/val/test data generation
def data_partition_leave_one_split(user_df):
    """create three user-item interaction dicts, the key is user ID, the value is item ID list"""
    User_data = {}
    User_label = {}
    user_train_data = {}
    user_valid_data = {}
    user_test_data = {}
    user_train_label = {}
    user_valid_label = {}
    user_test_label = {}
    
    if os.path.exists('user_data.pkl'):
        print(f'load data from file.')
        
        with open("user_data.pkl", "rb") as file:
            User_data = pickle.load(file)
        with open("user_label.pkl", "rb") as file:
            User_label = pickle.load(file)
    else:
        print(f'process data from stratch.')
        for user in tqdm(user_df['uid'].unique()):
            User_data[user] = user_df[user_df['uid'] == user][['iid', 'pros', 'cons', 'rating', 'category', 'pros_kw_list', 'cons_kw_list']]
            User_label[user] = User_data[user][['rating']]
            User_data[user] = User_data[user][['iid', 'pros', 'cons', 'category', 'pros_kw_list', 'cons_kw_list']]
            
        with open("user_data.pkl", "wb") as file:
            pickle.dump(User_data, file)
        with open("user_label.pkl", "wb") as file:
            pickle.dump(User_label, file)
        
    for user in tqdm(User_data):
        user_train_data[user] = User_data[user].iloc[:-2]
        user_train_label[user] = User_label[user].iloc[:-2]
        user_valid_data[user] = []
        user_valid_data[user].append(User_data[user].iloc[-2])
        user_valid_label[user] = []
        user_valid_label[user].append(User_label[user].iloc[-2])
        user_test_data[user] = []
        user_test_data[user].append(User_data[user].iloc[-1])
        user_test_label[user] = []
        user_test_label[user].append(User_label[user].iloc[-1])
    
    return user_train_data, user_train_label, user_valid_data, user_valid_label, user_test_data, user_test_label


def sample_function(user_train_data, user_train_label, uid_to_int, item_to_action, batch_size, maxlen, result_queue, SEED):
    def sample():
        """a randomly sampled user's training data, (userID, input sequence, labels, negative sequence), if the
         number of interactions is less than maxlen, the head missing positions are filled with 0"""
        user_ids = list(user_train_data.keys())
        user = random.choice(user_ids)
        
        interacted_items = user_train_data[user]
        interacted_lables = user_train_label[user]
        
        iids = interacted_items['iid'].tolist()
        labels = interacted_lables['rating'].tolist()
        
        seq_data = np.zeros([maxlen], dtype=np.int32)
        pos_data = np.zeros([maxlen], dtype=np.int32)
        pos_label = np.zeros([maxlen], dtype=np.float32)
        
        nxt_data = item_to_action[iids[-1]]
        nxt_label = labels[-1]
        
        idx = maxlen - 1
        for i in reversed(iids[:-1]):
            """seq is the input sequence"""
            seq_data[idx] = item_to_action[i]
            """pos is the next interaction of current interaction"""
            pos_data[idx] = nxt_data
            nxt_data = item_to_action[i]
            idx -= 1
            if idx == -1: break
        
        idx = maxlen - 1
        for i in reversed(labels[:-1]):
            pos_label[idx] = nxt_label
            nxt_label = i
            idx -= 1
            if idx == -1: break

        return (uid_to_int[user], seq_data, pos_data, pos_label)
    
    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            """construct a training batch whose element is a user's interaction history"""
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))
    
class WarpSampler(object):
    def __init__(self, user_train_data, user_train_label, uid_to_int, item_to_action, batch_size=64, maxlen=2, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(user_train_data,
                                                      user_train_label,
                                                      uid_to_int, 
                                                      item_to_action, 
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()
            
            
def get_embedding_dataset(user_set, data, label):
#     user_set = set(review_set['uid'])
    
#     hist_pros_item_idx = [index for index, val in enumerate(hist_item_label) if val == 1]
    emb_list = []
    label_list = []
    for u_ in tqdm(user_set):
        pros = data[u_]['pros'].values # (1151,)
        cons = data[u_]['cons'].values # (1151,)
        emb_list.append(np.concatenate((pros, cons)))
        label_list.append(label[u_])
    return emb_list, label_list



def process_valid_data_leave_one_split(user_train_data, user_valid_data, user_valid_label, maxlen, uid_to_int, item_to_action):

    user_ids = list(user_train_data.keys())
    
    user_indices, log_seqs, item_indices = [], [], []
    labels = []
    for u in user_ids:
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        
        filtered_rows = user_train_data[u]
        iids = filtered_rows['iid'].tolist() 
        
        for i in reversed(iids):
            seq[idx] = item_to_action[i]
            idx -= 1
            if idx == -1: break
        val_iid_df = user_valid_data[u][0]
        val_iid = val_iid_df['iid']
        item_idx = item_to_action[val_iid]
        user_indices.append(uid_to_int[u])
        log_seqs.append(seq)
        item_indices.append(item_idx)
        u_label = user_valid_label[u]
        labels.append(u_label[0])
    
    processed_data = [np.array(user_indices), np.array(log_seqs), np.array(item_indices)]
    return processed_data, labels


def process_test_data_leave_one_split(user_train_data, user_valid_data, user_test_data, user_test_label, maxlen, uid_to_int, item_to_action):
    user_ids = list(user_train_data.keys())
    user_indices, log_seqs, item_indices = [], [], []
    labels = []
    for u in user_ids:
        
        filtered_rows = user_train_data[u]
        iids = filtered_rows['iid'].tolist()  
        
        seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        """the validation sample is the input sequence of test phase"""
        seq[idx] = item_to_action[user_valid_data[u][0]['iid']]
        idx -= 1
        for i in reversed(iids):
            seq[idx] = item_to_action[i]
            idx -= 1
            if idx == -1: break
        item_idx = item_to_action[user_test_data[u][0]['iid']]
        user_indices.append(uid_to_int[u])
        log_seqs.append(seq)
        item_indices.append(item_idx)
        u_label = user_test_label[u]
        labels.append(u_label[0])
        
    processed_data = [np.array(user_indices), np.array(log_seqs), np.array(item_indices)]
    return processed_data, labels


class sasrec(nn.Module):
    def __init__(self, num_user, num_item):
        super().__init__()
        self.d_model = 32
        self.n_head = 1
        self.dropout_rate = 0
        self.maxlen = 5
        self.n_layer = 2
        self.user_embedding = nn.Embedding(num_embeddings=num_user, embedding_dim=self.d_model).cuda()
        self.item_embedding = nn.Embedding(num_embeddings=num_item, embedding_dim=self.d_model).cuda()
        self.pos_emb = nn.Embedding(self.maxlen, self.d_model).cuda()

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward = self.d_model, 
                                                   nhead=self.n_head, dropout = self.dropout_rate, batch_first = True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layer)
        self.affine_output1 = torch.nn.Linear(in_features=self.d_model*2, out_features=self.d_model)
        self.affine_output2 = torch.nn.Linear(in_features=self.d_model, out_features=1)
        self.out = nn.Sigmoid()
        
    def forward(self, uid, item_ids, cand_item_ids):

        user_emb = self.user_embedding(uid)
        item_emb = self.item_embedding(item_ids) # bs, max_len
        cand_item_emb = self.item_embedding(cand_item_ids) # bs, max_len, d_model
        
        batch_size, seq_len, _ = item_emb.size()
        position_indices = torch.arange(seq_len).unsqueeze(0).expand((batch_size, seq_len)).cuda()
        pos_encoding = self.pos_emb(position_indices)
        output = torch.add(item_emb, pos_encoding)
        output = self.transformer(output) # bs, max_len, d_model
#         print('{}, {}'.format(output.shape, user_emb.unsqueeze(1).repeat((1, seq_len, 1)).shape))
        output = torch.concat((output, user_emb.unsqueeze(1).repeat((1, seq_len, 1))), dim=-1)
        output = self.affine_output1(output)
        output = output * cand_item_emb
        output = self.affine_output2(output)
        output = self.out(output).squeeze(-1)
        return output
    
    def predict(self, uid, item_ids, cand_item_ids):
#         call for inference and output the last token
        uid = torch.from_numpy(uid).long().cuda()
        item_ids = torch.from_numpy(item_ids).long().cuda()
        cand_item_ids = torch.from_numpy(cand_item_ids).long().cuda()
        user_emb = self.user_embedding(uid).cuda()
        item_emb = self.item_embedding(item_ids).cuda()
        cand_item_emb = self.item_embedding(cand_item_ids).cuda()
        batch_size, seq_len, _ = item_emb.size()
        position_indices = torch.arange(seq_len).unsqueeze(0).expand((batch_size, seq_len)).cuda()

        pos_encoding = self.pos_emb(position_indices)

        assert item_emb.shape == pos_encoding.shape, '{} {}'.format(item_emb.shape, pos_encoding.shape)
        output = torch.add(item_emb, pos_encoding)
        output = self.transformer(output)
        
        output = torch.concat((output, user_emb.unsqueeze(1).repeat((1, seq_len, 1))), dim=-1)
        output = self.affine_output1(output)
        
        output = output[:,-1,:] * cand_item_emb
        output = self.affine_output2(output)
        output = self.out(output).squeeze(-1)

#         output = output[:,-1,:].matmul(cand_item_emb.T).unsqueeze(-1) # bs, ma
#         output = torch.bmm(output[:,-1,:].unsqueeze(1),cand_item_emb.unsqueeze(-1)) # bs, 1
        output = self.out(output)
        
        return output
        
    
# def __main__():
if __name__ == '__main__':
    batch_size = 512
    EPOCH = 1000
    # Logging.
#     base_dir = 'user_response_model/'
    base_dir = ''
    path = base_dir + 'log/'+'Yelp_MO'+'/'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logname = os.path.join(path, current_time+'.txt')
    initLogging(logname)

#     MO_fn = 'dataset/MO_dataset.pkl'
    MO_fn = base_dir + 'MO_dataset.pkl'
    with open(MO_fn, 'rb') as f:
        review_set = pickle.load(f)

    item_ids = review_set['iid'].unique()
    user_ids = review_set['uid'].unique()
    action_to_item = {}
    item_to_action = {}
    count = 0
    for id in item_ids:
        action_to_item[count] = id
        item_to_action[id] = count
        count += 1

    int_to_uid = {}
    uid_to_int = {}
    count = 0
    for id in user_ids:
        int_to_uid[count] = id
        uid_to_int[id] = count
        count += 1

    logging.info('User number: {}, item number: {}'.format(len(action_to_item.keys()), len(int_to_uid.keys())))
    logging.info('Split the data')
    user_train_data, user_train_label, user_valid_data, user_valid_label, user_test_data, user_test_label = data_partition_leave_one_split(review_set)


    sampler = WarpSampler(user_train_data, user_train_label, uid_to_int, item_to_action, batch_size=batch_size, maxlen=2, n_workers=2)
    logging.info('Process the evaluation data')
    processed_user_valid_data, processed_user_valid_label = process_valid_data_leave_one_split(user_train_data, user_valid_data, user_valid_label, 2, uid_to_int, item_to_action)
    processed_user_test_data, processed_user_test_label = process_test_data_leave_one_split(user_train_data, user_valid_data, user_test_data, user_test_label, 2, uid_to_int, item_to_action)
    num_batch = len(user_train_data) // batch_size

    model = sasrec(num_user=len(user_ids), num_item=len(item_ids))
    engine = Engine(model)
    
    save_path = base_dir + 'best_model.pth'
    val_auc_list = []
    test_auc_list = []
    val_logloss_list = []
    test_logloss_list = []
    val_mi_result_list = []
    test_mi_result_list = []
    val_ma_result_list = []
    test_ma_result_list = []
    val_we_result_list = []
    test_we_result_list = []
    best_val_auc = 0
    final_test_epoch = 0
    best_train_auc = 0
    for epoch in range(EPOCH):
    #     break
        logging.info('-' * 80)
        logging.info('Epoch {} starts !'.format(epoch))

        logging.info('-' * 80)
        logging.info('Training phase!')
        print('call train an epoch')
        engine.train_an_epoch(sampler, num_batch)
        # break

        logging.info('-' * 80)
        logging.info('Validating phase!')
        val_auc, val_logloss, val_mi_result, val_ma_result, val_we_result = engine.test(processed_user_valid_data, processed_user_valid_label)
        # break
        logging.info('[Validating Epoch {}] AUC = {:.4f}'.format(epoch, val_auc))
        val_auc_list.append(val_auc)
        logging.info('[Validating Round {}] LogLoss = {:.4f}'.format(epoch, val_logloss))
        val_logloss_list.append(val_logloss)
        logging.info('[Validating Round {}] Mi_Result = {}'.format(epoch, val_mi_result))
        val_mi_result_list.append(val_mi_result)
        logging.info('[Validating Round {}] Ma_Result = {}'.format(epoch, val_ma_result))
        val_ma_result_list.append(val_ma_result)
        logging.info('[Validating Round {}] We_Result = {}'.format(epoch, val_we_result))
        val_we_result_list.append(val_we_result)

        logging.info('-' * 80)
        logging.info('Testing phase!')
        test_auc, test_logloss, test_mi_result, test_ma_result, test_we_result = engine.test(processed_user_test_data, processed_user_test_label)
        logging.info('[Testing Epoch {}] AUC = {:.4f}'.format(epoch, test_auc))
        test_auc_list.append(test_auc)
        logging.info('[Testing Round {}] LogLoss = {:.4f}'.format(epoch, test_logloss))
        test_logloss_list.append(test_logloss)
        logging.info('[Testing Round {}] Mi_Result = {}'.format(epoch, test_mi_result))
        test_mi_result_list.append(test_mi_result)
        logging.info('[Testing Round {}] Ma_Result = {}'.format(epoch, test_ma_result))
        test_ma_result_list.append(test_ma_result)
        logging.info('[Testing Round {}] We_Result = {}'.format(epoch, test_we_result))
        test_we_result_list.append(test_we_result)

        if val_auc > best_val_auc:
            engine.save_para(save_path)
            best_val_auc = val_auc
            final_test_epoch = epoch


    engine.load_para(save_path)  

    logging.info('-' * 80)
    logging.info('Last testing phase!')
    test_auc, test_logloss, test_mi_result, test_ma_result, test_we_result = engine.test(processed_user_test_data, processed_user_test_label)
    logging.info('[Testing Epoch {}] AUC = {:.4f}'.format(epoch, test_auc))
    test_auc_list.append(test_auc)
    logging.info('[Testing Round {}] LogLoss = {:.4f}'.format(epoch, test_logloss))
    test_logloss_list.append(test_logloss)
    logging.info('[Testing Round {}] Mi_Result = {}'.format(epoch, test_mi_result))
    test_mi_result_list.append(test_mi_result)
    logging.info('[Testing Round {}] Ma_Result = {}'.format(epoch, test_ma_result))
    test_ma_result_list.append(test_ma_result)
    logging.info('[Testing Round {}] We_Result = {}'.format(epoch, test_we_result))
    test_we_result_list.append(test_we_result)
    sampler.close()


