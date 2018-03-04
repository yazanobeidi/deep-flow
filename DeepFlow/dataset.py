# ________                  _______________                
# ___  __ \____________________  ____/__  /________      __
# __  / / /  _ \  _ \__  __ \_  /_   __  /_  __ \_ | /| / /
# _  /_/ //  __/  __/_  /_/ /  __/   _  / / /_/ /_ |/ |/ / 
# /_____/ \___/\___/_  .___//_/      /_/  \____/____/|__/  
#                  /_/                                    
#
import pandas as pd
import numpy as np
from configparser import ConfigParser
import os
import glob

__author__ = 'yazan'

class Dataset(object):
    def __init__(self, dataset_name):
        print('Initializing DeepFlow Dataset: {}'.format(dataset_name))
        # Initialize configuration manager
        self.config = ConfigParser()
        self.config.read('config/config.cfg')
        self.name = dataset_name
        self.path = os.path.join(self.config.get('DeepFlow', 'path'), self.name)
        self.header = self._get_header()


    def _get_header(self):
        # First file with "*header*.csv" is used to define the file headers
        header_file_name = glob.glob(os.path.join(self.path,'*header*.csv'))[0]
        with open(os.path.join(self.path, header_file_name)) as f:
            header = f.readline().strip()
            return header

    def remove_missing_values(self):
        pass

    def normalize_features(self):
        pass

    def split