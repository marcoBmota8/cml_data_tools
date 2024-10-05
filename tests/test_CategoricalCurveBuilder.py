# Marco Barbero Mota
# First version: September 2023

import sys
sys.path.append('../cml_data_tools')
from cml_data_tools.curves import CategoricalCurveBuilder, RegressionCurveBuilder, build_patient_curves
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
            'channel': self.n_samples*['ANA pattern'],
            'value': np.random.choice(['smooth', 'speckle', 'nucleolar'], size=self.n_samples)
            })  # get the synthetic data dataframe

        # Generate data for another categorical lab
        data_dates_other = np.random.choice(all_dates, size=5, replace=False) # choose n_samples random dates for the lab changes
        data_other_df = pd.DataFrame(
            {
            'date': data_dates_other,
            'ptid': 5*['Rtest'],
            'mode': 3*['Categorical labs']+2*['Measurements'],
            'channel': 3*['other_lab']+2*['continuous_lab'],
            'value': np.hstack((np.random.choice(['val_1', 'val_2', ], size = 3), np.array([1.1, 2.56])))
            }) # get the dataframe to input into the builder
        
        # merge both datasets
        data_df = pd.concat([data_df, data_other_df], axis = 0)

        data_df.set_index('date', inplace=True)
        data_df.sort_index(inplace=True)
        data_df.reset_index(inplace=True)

        self.data_df = data_df
        self.all_channels = self.data_df['channel'].unique() # Get all channels in the data

        # Generate grid of timestamps
        self.grid = pd.date_range(start='2023-01-01', periods=30, freq=self.frequency)

        # Initiate the curve builder
        self.builder = CategoricalCurveBuilder(imputation_method=self.imputation_method)

        # Compute curves
        cat_data = data_df[data_df['mode']=='Categorical labs']
        self.curves_df = self.builder(data=cat_data.set_index('date'), grid=self.grid)
    
    def test1_curve_builder(self):
        # Mutually exclusive dummy curves ground truth
        curves_gt = pd.DataFrame(index = pd.date_range(start='2023-01-01', periods=30, freq='D'))
        curves_gt['other_lab val_1'] = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        curves_gt['other_lab val_2'] = np.ones((30,), dtype = int)-curves_gt['other_lab val_1']
        curves_gt['ANA pattern nucleolar'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
        curves_gt['ANA pattern smooth'] = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1])
        curves_gt['ANA pattern speckle'] = np.ones((30,), dtype = int)-curves_gt['ANA pattern nucleolar']-curves_gt['ANA pattern smooth']

        # Data that generates those curves with the 'nearest' imputation method
        data = {
            'ptid': 6*['Rtest'],
            'mode': 6*['Categorical labs'],
            'channel': ['other_lab', 'ANA pattern', 'other_lab', 'ANA pattern', 'ANA pattern', 'ANA pattern'],
            'value': ['val_2', 'speckle', 'val_1', 'nucleolar', 'smooth', 'smooth']
        }

        # Define the date index
        dates = ['2023-01-06', '2023-01-10', '2023-01-14', '2023-01-16', '2023-01-24', '2023-01-25']
        date_index = pd.to_datetime(dates)

        # Create the DataFrame
        df = pd.DataFrame(data, index=date_index)

        curves = self.builder(data = df, grid = self.grid)

        test_result = True if pd.testing.assert_frame_equal(curves_gt.sort_index(axis = 1), curves.sort_index(axis = 1), check_dtype=False)==None else False
        self.assertTrue(test_result, 'Test curve building NOT PASSED')
        print("Test curve building, PASSED")
        
    def test_mutual_exclusivity(self): 
        '''
        Test for temporal mutual exclusivity.
        '''
        for channel in self.all_channels:
            channel_curves_df = self.curves_df[self.curves_df.columns[self.curves_df.columns.str.contains(channel)]]
        if self.imputation_method in ['nearest', 'ffill']:
            self.assertTrue(np.array_equal(channel_curves_df.sum(axis =1).values, np.ones(len(channel_curves_df))), 'Mutual exclusivity test NOT PASSED')
            print('Mutual exclusivity test PASSED')
        elif self.imputation_method == 'bfill':
            last_value = self.data_df.dates.values[-1]
            self.assertTrue(np.array_equal(channel_curves_df.set_index('date').loc[last_value].sum(axis =1).values,\
                              np.ones(len(channel_curves_df.set_index('date').loc[last_value]))), 'Mutual exclusivity test NOT PASSED')
    
    def test_build_patient_curves(self):
        spec = {
            'Categorical labs': self.builder,
            'Measurements': RegressionCurveBuilder()
        }
        pat_curves = build_patient_curves(self.data_df, spec, resolution='D')
        self.assertTrue(pat_curves.columns.names==['mode', 'channel'], 'Wrong patient curves column multiindex')
        self.assertTrue(all(col_name in [
            'ANA pattern nucleolar',
            'ANA pattern speckle',
            'ANA pattern smooth',
            'other_lab val_1',
            'other_lab val_2',
            'continuous_lab'
            ] for col_name in pat_curves.columns.get_level_values(1) ), 'Wrong patient curves column names')
        
if __name__ == '__main__': 
    unittest.main()