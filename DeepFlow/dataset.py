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
from sklearn import preprocessing
from sklearn import model_selection
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

__author__ = 'yazan'

class Dataset(object):
    def __init__(self, dataset_name, read_conf={}, big_data=False,debug=False):
        """
        Params:
            dataset_name: project folder name in data directory (string)
            read_conf: kwargs for pd.csv_read() (dict)
            big_data: if true, entire dataset loaded into memory otherwise
                        data is always sampled. Warning: Not implemented yet.
        """
        print('Initializing DeepFlow Dataset: {}'.format(dataset_name))
        # Initialize configuration manager
        self.config = ConfigParser()
        self.config.read('config/config.cfg')
        self.name = dataset_name
        self.path = os.path.join(self.config.get('DeepFlow', 'path'), self.name)
        self.header = self._get_header()
        if big_data:
            # The complexity with this part is that all future algorithms may
            # be affected depending on how this is implemented. TODO.
            raise NotImplementedError
        else:
            self._data = self._load_complete_dataset(read_conf,debug)
        print("Finished loading dataset. Printing 1 example row:")
        print(self._data.iloc[0])

    def _get_header(self):
        """Returns the headers for the dataset. A file containing the headers
            in csv format must be present in the project data directory.
        """
        # First line of first "*header*.csv" is used to define the file headers
        header_f_name = glob.glob(os.path.join(self.path,'*header*.csv'))[0]
        try:
            with open(os.path.join(self.path, header_f_name)) as f:
                header = f.readline().strip().split(",")
                print("Found the following headers: {}".format(header))
                return header, header_f_name
        except:
            print("There is an issue with your header file. "\
                  "Please ensure a header.csv file is present in your "\
                  "project data directory. Your current configuration sets "\
                  "this as {}".format(self.config.get('DeepFlow', 'path')))
            raise

    def _load_complete_dataset(self, read_conf, debug):
        """Returns a pandas DataFrame object of the entire dataset.
        """
        _subsets = []
        data_dir = os.listdir(self.path)
        print("Found {} files, preparing raw dataset...".format(len(data_dir)))
        # sort the files in the directory in asccending numerical order,
        # if there is a number at the end of the file name
        data_dir.sort(key=lambda x: '{0:0>256}'.format(x).lower())
        for f_name in data_dir:
            if f_name.endswith(".csv") and f_name is not self.header[1]:
                _subset = pd.read_csv(os.path.join(self.path,f_name),
                                      names=self.header[0], **read_conf)
                _subsets.append(_subset)
                # Clear so if read_csv fails silently we still catch it
                _subset = None
                if debug:
                    break
                print("\rstill processing ...")
        dataframe = pd.concat(_subsets, ignore_index=True)
        # Remove any rows that match the header
        for head in self.header[0]:
            try:
                dataframe = dataframe[dataframe[str(head)] != head]
            except TypeError:
                pass
        return dataframe

    def data(self):
        return self._data

    def remove_missing_values(self, dataset):
        """Returns a new pd.DataFrame with rows with NaN values removed.
        """
        print("{} NaN values in dataset".format(np.sum(dataset.isna().values)))
        return dataset.dropna()

    def normalize_features(self, dataset):
        """Returns a new pd.DataFrame where each column is normalized.
        TODO: allow user to specify columns to normalize. Need this anyways
        because not all columns are floats.
        """
        return dataset
        #print("Normalizing all rows.")
        #x = dataset.values
        #normalizer = preprocessing.MinMaxScaler()
        #x_norm = normalizer.fit_transform(x)
        #return pd.DataFrame(x_norm, header=self.header[0])

    def split_train_test_validation(self, 
                                    dataset, 
                                    test_size=0.33,
                                    validation_size=0.15):
        _train, test = model_selection.train_test_split(dataset,
                                                       test_size=test_size, 
                                                       random_state=436)
        train, validation = model_selection.train_test_split(_train,
                                                       test_size=validation_size, 
                                                       random_state=436)
        return train, test, validation

    def random_sample(self, dataset, percent):
        """Returns a new pd.DataFrame of a randomized sample of rows.
        TODO
        """
        return dataset
        #print(dataset.shape[0])
        #size = dataset.shape[0] * percent
        #random_rows = np.random.choice(dataset.values,(size,dataset.shape[1:]))
        #return pd.DataFrame(random_rows, header=self.header[0])

    def select_features(self):
        """Returns a new pd.DataFrame with only the specified columns.
        """
        pass

    def randomize(self):
        pass