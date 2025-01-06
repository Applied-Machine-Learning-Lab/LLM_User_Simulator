import json
from environment.yelp.yelp import Yelp
import pandas as pd
import pickle
from tqdm import tqdm


class YelpLoader():

    def __init__(self, ):
        with open('dataset/MO_dataset.pkl', 'rb') as f:
            self.data = pickle.load(f)

    def load_items_from_ids(self, id_):

        data = self.data[self.data['iid']==id_].iloc[0]
        movie = Yelp(data['uid'], data['iid'], data['pros'], data['cons'], data['rating'], data['date'], data['category'], data['pros_kw_list'], data['cons_kw_list'])
        return movie

    
    def load_item_from_id_dataframe(self, iid):
        data = self.data[self.data['iid']==iid].iloc[0]
        
        return data

    def load_interacted_items_class(self, uid, curr_iid, date):
        interacted_items = []
        filtered_df = self.data[(self.data['uid'] == uid) & (self.data['iid'] != curr_iid) & (self.data['date'] < date)] 
        if len(filtered_df) == 0:
            return None
        
        for index, row in filtered_df.iterrows():
            interacted_items.append(Yelp(row['uid'], row['iid'], row['pros'], row['cons'], row['rating'], row['date'], row['category'], row['pros_kw_list'], row['cons_kw_list']))
        return interacted_items
    

    def load_interacted_items(self, uid, curr_iid, date):
        interacted_items = []
        if curr_iid is not None:
            filtered_df = self.data[(self.data['uid'] == uid) & (self.data['iid'] != curr_iid) & (self.data['date'] < date)] 
        else:
            filtered_df = self.data[self.data['uid'] == uid]
            
        if len(filtered_df) == 0:
            return None, None
        
        for index, row in filtered_df.iterrows():
            data_ = {'uid': row['uid'], 
                     'iid': row['iid'], 
                     'pros': row['pros'], 
                     'cons': row['cons'], 
                     'rating': row['rating'], 
                     'date': row['date'], 
                     'category': row['category'], 
                     'pros_kw_list': row['pros_kw_list'], 
                     'cons_kw_list': row['cons_kw_list']
                    }
            interacted_items.append(data_)
    
        interacted_items = pd.DataFrame(interacted_items)
        return interacted_items[['iid', 'pros', 'cons', 'category', 'pros_kw_list', 'cons_kw_list']], interacted_items[['rating']]
    
    
    
    
    
    
    
    