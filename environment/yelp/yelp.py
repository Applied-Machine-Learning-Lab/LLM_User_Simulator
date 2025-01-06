from environment.item import Item

class Yelp(Item):

    def __init__(
        self,
        uid,
        iid,
        pros,
        cons,
        rating,
        date,
        category,
        pros_kw_list,
        cons_kw_list
    ):
        self.uid = uid
        self.iid = iid
        self.pros = pros
        self.cons = cons
        self.rating = rating
        self.date = date
        self.category = category
        self.pros_kw_list = pros_kw_list
        self.cons_kw_list = cons_kw_list

    @staticmethod
    def from_json(iid):
        data = self.data[self.data['iid']==iid]

        return Yelp(
            data["uid"],
            data["iid"],
            data["pros"],
            data["cons"],
            data["rating"],
            data["date"]
        )
    
    
