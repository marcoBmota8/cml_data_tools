# Marco Barbero Mota
# First version: September 2024

import sys
sys.path.append('../cml_data_tools')
from cml_data_tools.curves import ValueCurveBuilder, RegressionCurveBuilder, build_patient_curves
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
            'mode': 5*['Measurements'],
            'channel': ['lab1', 'lab1', 'lab2', 'lab2', 'lab2'],
            'value': [1, 2, 1, 2, 1]
            })  # get the synthetic data dataframe

        # Generate data for another categorical lab
        data_dates_other = np.random.choice(all_dates, size=5, replace=False) # choose n_samples random dates for the lab changes
        data_other_df = pd.DataFrame(
            {
            'date': data_dates_other,
            'ptid': 5*['Rtest'],
            'mode': 5*['BMI'],
            'channel': 5*['bmi'],
            'value': [23,23,21,20,25]
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
        self.builder = ValueCurveBuilder(imputation_method=self.imputation_method)
    
    def test1_curve_builder(self):
        # Mutually exclusive dummy curves ground truth
        curves_gt = pd.DataFrame(index = pd.date_range(start='2023-01-01', periods=30, freq='D'))
        curves_gt['lab1'] = np.array([1,1,1,1,1,1,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,2,2,2,2,2,2,2,2,2,2])
        curves_gt['lab2'] = np.array([10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,20,20,20,20,20,20,20,20,20,20])

        # Data that generates those curves with the 'nearest' imputation method
        data = {
            'ptid': 6*['Rtest'],
            'mode': 6*['Measurements'],
            'channel': ['lab1', 'lab1', 'lab1', 'lab2', 'lab2', 'lab2'],
            'value': [1, 0.5, 2, 10, 2, 20]
        }

        # Define the date index
        dates = ['2023-01-06', '2023-01-16', '2023-01-25', '2023-01-16', '2023-01-25', '2023-01-25']
        date_index = pd.to_datetime(dates)

        # Create the DataFrame
        df = pd.DataFrame(data, index=date_index)

        curves = self.builder(data = df, grid = self.grid)

        test_result = True if pd.testing.assert_frame_equal(curves_gt.sort_index(axis = 1), curves.sort_index(axis = 1),
                                                            check_dtype=False,
                                                            check_freq=False,
                                                            check_names=False)==None else False
        self.assertTrue(test_result, 'Test curve building NOT PASSED')
        print("Test curve building PASSED")
        
    def test_build_patient_curves(self):
        spec = {
            'BMI': RegressionCurveBuilder(),
            'Measurements': self.builder
        }
        pat_curves = build_patient_curves(self.data_df, spec, resolution='D')
        self.assertTrue(pat_curves.columns.names==['mode', 'channel'], 'Wrong patient curves column multiindex')
        self.assertTrue(all(col_name in ['bmi', 'lab1', 'lab2'] for col_name in pat_curves.columns.get_level_values(1) ), 'Wrong patient curves column names')
        print('build_patient_curves test PASSED')
        
if __name__ == '__main__': 
    unittest.main()