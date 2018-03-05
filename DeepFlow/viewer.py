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

__author__ = 'yazan'

class View(object):
    def __init__(self, dataset_object):
        print('Initializing Deepflow Viewer: {}'.format(dataset_object.name))
        self.dataset = dataset_object

    def table(self, df):
        return df

    def plot(self, df, x, y=None):
        #dfy = df.loc[:, df.columns != str(x_label)]
        df.plot(x=x, y=y, figsize=(10,8))

    def histogram(self):
        pass

    def summary_statistics(self):
        pass

    def clusters(self):
        pass