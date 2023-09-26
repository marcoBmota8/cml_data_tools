# Marco Barbero Mota
# First version: September 2023

import sys
sys.path.append('../cml_data_tools')
from curves import CategoricalCurveBuilder
import pandas as pd
import numpy as np
import unittest
from pandas.testing import assert_frame_equal

class TestCategoricalCurvebuilder(unittest.TestCase):

    def setUp(self):
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
        self.builder = CategoricalCurveBuilder(imputation_method=self.imputation_method)

        # Compute curves
        self.curves_df = self.builder(data=self.data_df, grid=self.grid)
    
    def test1_curve_builder(self):
        # Mutually exclusive dummy curves ground truth
        curves_gt = pd.DataFrame(index = pd.date_range(start='2023-01-01', periods=30, freq='D'))
        curves_gt['other_lab_val_1'] = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        curves_gt['other_lab_val_2'] = np.ones((30,), dtype = int)-curves_gt['other_lab_val_1']
        curves_gt['ANA_pattern_nucleolar'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
        curves_gt['ANA_pattern_smooth'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1])
        curves_gt['ANA_pattern_speckle'] = np.ones((30,), dtype = int)-curves_gt['ANA_pattern_nucleolar']-curves_gt['ANA_pattern_smooth']

        # Data that generates those curves with the 'nearest' imputation method
        data = {
            'ptid': 6*['Rtest0'],
            'mode': 6*['Categorical labs'],
            'channel': ['other_lab', 'ANA_pattern', 'other_lab', 'ANA_pattern', 'ANA_pattern', 'ANA_pattern'],
            'value': ['val_2', 'speckle', 'val_1', 'nucleolar', 'smooth', 'smooth']
        }

        # Define the date index
        dates = ['2023-01-06', '2023-01-10', '2023-01-14', '2023-01-16', '2023-01-24', '2023-01-25']
        date_index = pd.to_datetime(dates)

        # Create the DataFrame
        df = pd.DataFrame(data, index=date_index)

        curves = self.builder(data = df, grid = self.grid)

        assert_frame_equal(curves_gt.sort_index(axis = 1), curves.sort_index(axis = 1), check_dtype=False)
        
    def test_mutual_exclusivity(self): 
        '''
        Test for temporal mutual exclusivity.
        '''
        for channel in self.all_channels:
            channel_curves_df = self.curves_df[self.curves_df.columns[self.curves_df.columns.str.contains(channel)]]
        if self.imputation_method in ['nearest', 'ffill']:
            self.assertTrue(np.array_equal(channel_curves_df.sum(axis =1).values, np.ones(len(channel_curves_df))), 'Mutual exclusivity test NOT PASSED')
            print('test PASSED')
        elif self.imputation_method == 'bfill':
            last_value = self.data_df.dates.values[-1]
            self.assertTrue(np.array_equal(channel_curves_df.set_index('dates').loc[last_value].sum(axis =1).values,\
                              np.ones(len(channel_curves_df.set_index('dates').loc[last_value]))), 'Mutual exclusivity test NOT PASSED')
            print('test PASSED')
     
if __name__ == '__main__': 
    unittest.main()