# Marco Barbero Mota
# First version: 25th September 2023
# %%

import sys
sys.path.append('/home/barbem4/cml_data_tools/cml_data_tools')
from curves import CategoricalCurveBuilder,BinaryCurveBuilder,build_patient_curves
import pandas as pd
import numpy as np
import unittest

# %%
class TestCategoricalCurvebuilder(unittest.TestCase):

   def setup(self):
        
        # test parameters
        self.imputation_method = 'nearest'
        self.n_samples = 5
        self.frequency = 'D'

        # Generate random data
        # get all january dates
        all_dates = pd.date_range(
            start='2023-01-01', periods=30, freq=self.frequency)
        # choose n_samples random dates for the lab changes
        data_dates = np.random.choice(all_dates, size=self.n_samples, replace=False)
        data_df = pd.DataFrame(
            {
            'date': data_dates,
            'ptid': self.n_samples*['Rtest'],
            'mode': self.n_samples*['Categorical labs'],
            'channel': self.n_samples*['ANA_pattern'],
            'value': np.random.choice(['smooth', 'speckle', 'nucleolar'], size=self.n_samples)
            })  # get the synthetic data dataframe

        # Generate data for another categorical lab
        data_dates_other = np.random.choice(all_dates, size=3, replace=False) # choose n_samples random dates for the lab changes
        data_other_df = pd.DataFrame(
            {
            'date': data_dates_other,
            'ptid': 3*['Rtest0'],
            'mode': 3*['Categorical labs'],
            'channel': 3*['other_lab'],
            'value': np.random.choice(['val_1', 'val_2', ], size = 3)
            }) # get the dataframe to input into the builder
        
        # merge both datasets
        data_df = pd.concat([data_df, data_other_df], axis = 0)

        data_df.set_index('date', inplace=True)
        data_df.sort_index(inplace=True)

        self.data_df = data_df
        self.all_channels = self.data_df['channel'].unique() # Get all channels in the data

        # Generate grid of timestamps
        self.grid = pd.date_range(start='2023-01-01', periods=30, freq=self.frequency)

        # Initiate the curve builder
        builder = CategoricalCurveBuilder(imputation_method=self.imputation_method)

        # Compute curves
        self.curves_df = builder(data=self.data_df, grid=self.grid)

    def test_curve_builder(self):
        # TODO make a test where we know the ground truth of the curves and compare against the builder output
    
    def test_mutual_exclusivity(self): 
        '''
        Test for temporal mutual exclusivity.
        '''
        for channel in  #TODO -> NEEDS to iterate over channels from self.all_channels
        # adn to consider to only sum those colmns that have the prefix of the channel
        if self.imputation_method in ['nearest', 'ffill']:
            self.assertEqual(self.curves_df.sum(axis =1).values, np.ones(len(self.curves_df))).all(), 'Mutual exclusivity test NOT PASSED'
            print('test PASSED')
        elif self.imputation_method == 'bfill':
            last_value = self.data_df.dates.values[-1]
            self.assertEqual(self.curves_df.set_index('dates').loc[last_value].sum(axis =1).values,\
                              np.ones(len(self.curves_df.set_index('dates').loc[last_value]))).all(), 'Mutual exclusivity test NOT PASSED'
            print('test PASSED')
     

# %%
if __name__ == '__main__': 
    unittest.main()




# # Generate random data
# all_dates = pd.date_range(start='2023-01-01', periods=30, freq='D') #get all january dates
# data_dates = np.random.choice(all_dates, size=5, replace=False) # choose n_samples random dates for the lab changes
# data_df = pd.DataFrame(
#     {
#     'date': data_dates,
#     'ptid': 5*['Rtest0'],
#     'mode': 5*['Categorical labs'],
#     'channel': 5*['ANA_pattern'],
#     'value': np.random.choice(['smooth', 'speckle', 'nucleolar'], size = 5)
#     }) # get the dataframe to input into the builder

# data_dates_other = np.random.choice(all_dates, size=3, replace=False) # choose n_samples random dates for the lab changes
# data_other_df = pd.DataFrame(
#     {
#     'date': data_dates_other,
#     'ptid': 3*['Rtest0'],
#     'mode': 3*['Categorical labs'],
#     'channel': 3*['other_lab'],
#     'value': np.random.choice(['val_1', 'val_2', ], size = 3)
#     }) # get the dataframe to input into the builder

# # merge both datasets
# data_df = pd.concat([data_df, data_other_df], axis = 0)

# data_df.set_index('date', inplace = True) # set date as DatatimeIndex

# grid = pd.date_range(start='2023-01-01', periods=30, freq='D')

# builder = CategoricalCurveBuilder(imputation_method='nearest')
# curves_df = builder(data = data_df, grid = grid)




# THIS SEEMED LIKE A GOOD IDEA BECAUSE IT USES A VALIDATED BUILDER TO BUILD THE SAME CURVES
# but the BinaryCurveBuilder does not consider mutual exclusivity so the curves would be wrong

    # # Build curves sequentially using BinaryCurveBuilder
    # builder_binary = BinaryCurveBuilder(imputation_method=self.imputation_method)
    # # Build each curve sequentially
    # curves_gt = pd.DataFrame(index = self.grid)
    # all_channels = self.data_df['channel'].unique() # Get all channels in the data
    # for channel in all_channels:
    #     data_channel = self.data_df[self.data_df['channel']==channel] # get the data for each channel
    #     for result in data_channel['value'].unique():
    #         col_name = data_channel['channel'].unique()[0]+'_'+result # column name for each test result
    #         curves_gt[col_name] = builder_binary(data = data_channel, grid = self.grid) # append the column recently computed
                
