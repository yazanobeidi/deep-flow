# ________                  _______________                
# ___  __ \____________________  ____/__  /________      __
# __  / / /  _ \  _ \__  __ \_  /_   __  /_  __ \_ | /| / /
# _  /_/ //  __/  __/_  /_/ /  __/   _  / / /_/ /_ |/ |/ / 
# /_____/ \___/\___/_  .___//_/      /_/  \____/____/|__/  
#                  /_/                                    
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

__author__ = 'yazan'

class View(object):
    def __init__(self, dataset_object):
        print('Initializing Deepflow Viewer: {}'.format(dataset_object.name))
        self.dataset = dataset_object

    def table(self, df):
        return df

    def plot(self, df, x=None, y=None, title=None, reindex_like=None, logx=False):
        if type(df) == type(list()):
            reindex = df[0] if reindex_like is None else reindex_like
            df = pd.concat([d.reset_index(drop=True) for d in df], axis=1)
        x_label = df.index if x is None else x
        df.plot(x=x_label, y=y, figsize=(10,8), title=title, logx=logx)

    def histogram(self):
        pass

    def summary_statistics(self):
        pass

    def clusters(self):
        pass