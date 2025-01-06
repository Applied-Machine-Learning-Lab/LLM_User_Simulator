import os
import string
from functools import reduce

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

from environment.item import ItemsLoader
from environment.items_selection import ItemsSelector
from user_response_model.SASRec import sasrec
from user_response_model.engine import Engine
import random
from sklearn.metrics.pairwise import cosine_similarity as cos_
from sklearn.metrics.pairwise import euclidean_distances as l2_


cnt_all_pros = 0
cnt_all_cons = 0
cnt_oths = 0
    
def infer_user_interaction(hist_item, hist_item_label, cand_item, sim='cos'):
    '''
    infer whether user interact or not given the candidate item and historical interactions, including keywords matching and keywords embedding comparison
    hist_item: history item list, each item contains iid, pros embedding, cons embedding
    cand_item: iid, pros embedding, cons embedding
    sim: similarity metric, cos: cosine similarity; l2: l2 distance
    
    Return: candidate item inference
    '''
    global cnt_all_pros
    global cnt_all_cons
    global cnt_oths
    
    hist_pros_item_idx = [index for index, val in enumerate(list(hist_item_label['rating'])) if val == 1]
    if len(hist_pros_item_idx) == 0:
        cnt_all_pros += 1
        return 1, 1
    hist_pros_item_list = [hist_item.iloc[i]['pros'] for i in hist_pros_item_idx]
    
    hist_cons_item_idx = [index for index, val in enumerate(list(hist_item_label['rating'])) if val == 0]
    if len(hist_cons_item_idx) == 0:
        cnt_all_cons += 1
        return 0, 0
    hist_cons_item_list = [hist_item.iloc[i]['cons'] for i in hist_cons_item_idx]
    
    cnt_oths += 1
    
    str_to_set = lambda s: {keyword.strip() for keyword in s.split(',')} if s==s else set()
    cand_item_cate = str_to_set(cand_item['category'])
    
    hist_pros_cate_idx = [i for i in hist_pros_item_idx if str_to_set(hist_item.iloc[i]['category']) & cand_item_cate]
    hist_pros_cate = [hist_item.iloc[i]['pros'] for i in hist_pros_cate_idx]

    
    hist_cons_cate_idx = [i for i in hist_cons_item_idx if str_to_set(hist_item.iloc[i]['category']) & cand_item_cate]
    hist_cons_cate = [hist_item.iloc[i]['cons'] for i in hist_cons_cate_idx]
        
        
    def check_kw_match(cand_item, hist_pros_kw_list, hist_cons_kw_list):
        '''
        cand_item: candidate item, a piece of dataframe, need its pros kw list and cons kw list
        hist_pros_item_list: list of (pros kw list)
        hist_cons_item_list: list of (cons kw list)
        '''
        def str_to_set(s):
            kw_list = s.strip("[]").split(',')
            return {kw.strip("' ") for kw in kw_list}
        
        pros_match_num = cons_match_num = 0
        cand_item_pros_kw_set = str_to_set(cand_item['pros_kw_list'])
        cand_item_cons_kw_set = str_to_set(cand_item['cons_kw_list'])
        for pros_ in hist_pros_kw_list:
            pros_match_num += len(cand_item_pros_kw_set & str_to_set(pros_))
        for cons_ in hist_cons_kw_list:
            cons_match_num += len(cand_item_cons_kw_set & str_to_set(cons_))
        
        return pros_match_num, cons_match_num
        
    def cal_similary(cand_item, hist_pros_item_list, hist_cons_item_list, sim=sim):
        '''
        cand_item: candidate item, a piece of dataframe, need its pros and cons embedding
        hist_pros_item_list: list of pros embedding 
        hist_cons_item_list: list of cons embedding
        '''
        
        cosine_sim_pros = [cos_(cand_item['pros'].reshape(1, -1), hist_pros.reshape(1, -1))[0, 0] for hist_pros in hist_pros_item_list]
        max_sim_pros = max(cosine_sim_pros)
        cosine_sim_cons = [cos_(cand_item['cons'].reshape(1, -1), hist_cons.reshape(1, -1))[0, 0] for hist_cons in hist_cons_item_list]
        max_sim_cons = max(cosine_sim_cons)
        
        return max_sim_pros, max_sim_cons
    
    
#     If there are similar products in historical interactions->calculate the number of keyword matches with similar products; if there are no similar products->calculate the number of keyword matches for all products
    hist_pro_kw_list = hist_con_kw_list = None
    if len(hist_pros_cate_idx) > 0:
        hist_pro_kw_list = [hist_item.iloc[i]['pros_kw_list'] for i in hist_pros_cate_idx]
    else:
        hist_pro_kw_list = [hist_item.iloc[i]['pros_kw_list'] for i in hist_pros_item_idx]
    if len(hist_cons_cate_idx) > 0:
        hist_con_kw_list = [hist_item.iloc[i]['cons_kw_list'] for i in hist_cons_cate_idx]
    else:
        hist_con_kw_list = [hist_item.iloc[i]['cons_kw_list'] for i in hist_cons_item_idx]
    
    max_sim_pros_matching, max_sim_cons_matching = check_kw_match(cand_item, hist_pro_kw_list, hist_con_kw_list)
    
    
#     Calculate similarity. If there are similar products in the historical interaction, -> calculate the similarity with similar products. If there are no similar products, -> calculate the similarity of all products.
    hist_pro_ = hist_con_ = None
    if len(hist_pros_cate_idx) > 0:
        hist_pro_ = hist_pros_cate
    else:
        hist_pro_ = hist_pros_item_list
    if len(hist_cons_cate_idx) > 0:
        hist_con_ = hist_cons_cate
    else:
        hist_con_ = hist_cons_item_list
    
    max_sim_pros_embedding, max_sim_cons_embedding = cal_similary(cand_item, hist_pro_, hist_con_, sim)
    
    return max_sim_pros_matching > max_sim_cons_matching, max_sim_pros_embedding > max_sim_cons_embedding

class UserSim(gym.Env):
    def __init__(
        self,
        render_mode: str,
        items_loader: ItemsLoader,
        items_selector: ItemsSelector,
        render_path: str = "./tmp/render/",
        evaluation: bool = False,
    ):
        self.render_mode = render_mode
        self.render_path = render_path
        self.metadata = {"render_modes": ["human", "csv"]}
        
        self.max_len = 2
        self.user_list = items_loader.data['uid'].unique()
        self.num_users = len(self.user_list)

        self.int_to_uid = {}
        self.uid_to_int = {}
        count = 0
        for id in self.user_list:
            self.int_to_uid[count] = id
            self.uid_to_int[id] = count
            count += 1

        """
        Initialize the item loader
        """
        self.items_loader = items_loader
        self.item_ids = items_loader.data['iid'].unique()
        self.num_items = len(self.item_ids)
        
        self.model = sasrec(num_user=self.num_users, num_item=self.num_items)
        self.engine = Engine(self.model)
        self.engine.load_para(save_path='user_response_model/best_model.pth')

        self.observation_space = spaces.Dict(
            {
                "user_id": spaces.Discrete(self.num_users),
                "items_interact": spaces.Sequence(
                    spaces.Box(
                        low=np.array([0, 0]),
                        high=np.array([self.num_items, 11]),
                        shape=(2,),
                        dtype=np.int_,
                    )
                ),
            }
        )
        
        self.action_space = spaces.Discrete(self.num_items)

        self.action_to_item = {}
        self.item_to_action = {}
        count = 0
        for id in self.item_ids:
            self.action_to_item[count] = id
            self.item_to_action[id] = count
            count += 1


        self.items_selector = items_selector

        self.evaluation = evaluation
        self.evaluation_previous_user_id = None
        self.evaluation_count = 0
        
        self.tmr_thre = 0.5

    def _get_obs(self):
#         gender = 0 if self._user.gender == "M" else 1
        return {
            "user_id": self._user,
            "items_interact": self._items_interact,
        }

    
    def get_inter_items_rl(self, _items_interact):
        '''
        input type: tuple()

        self._items_interact = self._items_interact + (
            np.array(
                [self.item_to_action[item_id], reward],
                dtype=np.int_,
            ),
        )
        '''
        if len(_items_interact) == 0:
            return None, None


        interacted_items = pd.DataFrame(columns = self.items_loader.data.columns)
        interacted_items_labels = []
        for l_ in _items_interact:
            l_id = self.action_to_item[l_[0]]
            data_ = self.items_loader.load_item_from_id_dataframe(l_id)
            interacted_items.loc[len(interacted_items)] = data_
            interacted_items_labels.append(l_[1])
        return interacted_items, interacted_items_labels

    def reset(self, seed=None, options=None, user_id=None):
        """
        The reset function resets the environment, this is done by selecting a new user
        the user selection is performed at random if the user_id input is None
        """
        super().reset(seed=seed)
        if seed is not None:
            self.items_selector.seed(seed)

        """
        Initialize a new user by picking a random user id between 0 and num_users if user_id is None
        """
        if user_id is None and not self.evaluation:
            user_id = self.np_random.integers(low=0, high=self.num_users)
        elif self.evaluation:
            if self.evaluation_previous_user_id is None:
                user_id = 0
                self.evaluation_count = 0
            else:
                user_id = self.evaluation_previous_user_id + 1
                self.evaluation_count = 0
        self._user = user_id

        inter_items, inter_items_labels = self.items_loader.load_interacted_items(self.int_to_uid[user_id], None, None)
    
        self._items_interact = tuple()
        for i in range(self.max_len):
            self._items_interact = self._items_interact + (
                np.array(
                    [self.item_to_action[inter_items.iloc[i]['iid']], inter_items_labels.iloc[i]['rating']],
                    dtype=np.int_,
                ),
            )
            
        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action: int):
        item_id = self.action_to_item[action]

        curr_item = self.items_loader.load_items_from_ids(item_id)
        _iid = curr_item.iid
        _uid = curr_item.uid
        _date = curr_item.date

        inter_items, inter_items_labels = self.items_loader.load_interacted_items(curr_item.uid, curr_item.iid, curr_item.date)
        inter_items_rl, inter_items_labels_rl = self.get_inter_items_rl(self._items_interact)
        
        if inter_items is None:
            reward = random.randint(0, 1)
        else:
            if inter_items_rl is None:
                inter_items_full = inter_items
                inter_items_labels_full = inter_items_labels
            else:
                inter_items_full = pd.concat([inter_items, inter_items_rl], axis=0)
                inter_items_labels_full = pd.concat([inter_items_labels, pd.DataFrame(inter_items_labels_rl, columns=inter_items_labels.columns)], axis=0)
                assert len(inter_items_full) == len(inter_items_labels_full), f'items: {len(inter_items_full)}, labels: {len(inter_items_labels_full)}'
                
            reward_matching, reward_embedding = infer_user_interaction(inter_items_full, inter_items_labels_full, {'pros': curr_item.pros,'cons': curr_item.cons, 'category': curr_item.category, 'pros_kw_list': curr_item.pros_kw_list, 'cons_kw_list': curr_item.cons_kw_list})
            reward_matching = 1 if reward_matching else 0
            reward_embedding = 1 if reward_embedding else 0
            
            
#             call trained SASRec and get test prediction
            tmr_user_id = np.array([self.uid_to_int[curr_item.uid]])
            seq = np.zeros([self.max_len], dtype=np.int32)
            idx = self.max_len - 1
            
            for item in reversed(list(inter_items_full['iid'])):
                seq[idx] = self.item_to_action[item]
                idx -= 1
                if idx == -1: break

            tmr_hist_item_list = np.array([seq])
            tmr_cand_item = np.array([self.item_to_action[curr_item.iid]])
            tmr_cand_item_label = None

            tmr_label = self.engine.infer([tmr_user_id, tmr_hist_item_list, tmr_cand_item], tmr_cand_item_label)
            reward_tmr = 1 if tmr_label >=self.tmr_thre else 0
        
            reward = 1 if reward_matching + reward_embedding + reward_tmr >= 2 else 0
            
        self._items_interact = self._items_interact + (
            np.array(
                [self.item_to_action[item_id], reward],
                dtype=np.int_,
            ),
        )
        # print('{}, {}, {}, total reward: {}'.format(reward_matching, reward_embedding, reward_tmr, reward))
        
        terminated = self.np_random.choice([True, False], p=[0.025, 0.975])
        terminated = bool(terminated)
        observation = self._get_obs()
        info = {
            "rating": reward,
        }

        # Handles evaluation termination
        if self.evaluation:
            self.evaluation_previous_user_id = self._user.id
            self.evaluation_count += 1
            terminated = False
            if self.evaluation_count == 11:
                terminated = True

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"User: {self._user.name}, List of interacted items:"
                f" {self._items_interact[:-1]}"
                + (
                    f" Item proposed: {self._items_interact[-1][0]}, User reward:"
                    f" {self._items_interact[-1][1]}  "
                    if len(self._items_interact) > 0
                    else ""
                )
            )
        if len(self._items_interact) > 0 and self.render_mode == "csv":
            df = pd.DataFrame(
                {
                    "user_id": [self._user.id],
                    "user_name": [self._user.name],
                    "time": [len(self._items_interact)],
                    "movie_id": np.stack(list(self._items_interact))[-1, 0],
                    "rating": np.stack(list(self._items_interact))[-1, 1],
                },
            )

            if os.path.exists(self.render_path):
                df.to_csv(
                    self.render_path,
                    mode="a",
                    index=False,
                    header=False,
                )
            else:
                os.makedirs(os.path.dirname(self.render_path), exist_ok=True)
                df.to_csv(
                    self.render_path,
                    index=False,
                )

