# ________                  _______________                
# ___  __ \____________________  ____/__  /________      __
# __  / / /  _ \  _ \__  __ \_  /_   __  /_  __ \_ | /| / /
# _  /_/ //  __/  __/_  /_/ /  __/   _  / / /_/ /_ |/ |/ / 
# /_____/ \___/\___/_  .___//_/      /_/  \____/____/|__/  
#                  /_/                                    
#

import pandas as pd
import numpy as np

__author__ = 'yazan'

class DeepFlow(object):
    def __init__(self, dataset_object):
        print('Initializing DeepFlow: {}'.format(dataset_object.name))

    def convnet(self, mode, param=None):
        pass

    def feedforward(self, mode, param=None):
        pass

    def rnn(self, mode, param=None):
        pass