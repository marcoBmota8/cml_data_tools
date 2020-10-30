#!/usr/bin/env python
"""
Example application using ICA
"""
import logging
from pathlib import Path

import sd_access
from cml_data_tools.config import Config
from cml_data_tools.curves import *
from cml_data_tools.experiment import Experiment
from cml_data_tools.pickle_cache import PickleCache
from cml_data_tools.source_ehr import DataSource, MetaSource
from cml_data_tools.standardizers import *


configs = [
    Config(
        mode='Measurements',
        data=DataSource('cml_test_measurement', sd_access.connect),
        meta=MetaSource('cml_test_measurement_meta', sd_access.connect),
        curve_cls=RegressionCurveBuilder,
        std_cls=LogGelmanStandardizerWithFallbacks,
        std_kws={'eps': 1e-6},
    ),
    Config(
        mode='Conditions',
        data=DataSource('cml_test_condition', sd_access.connect),
        meta=MetaSource('cml_test_condition_meta', sd_access.connect),
        curve_cls=IntensityCurveBuilder,
        std_cls=LogGelmanStandardizerWithFallbacks,
        std_kws={'eps': 1e-6},
    ),
    Config(
        mode='Medications',
        data=DataSource('cml_test_medication', sd_access.connect),
        meta=MetaSource('cml_test_medication_meta', sd_access.connect),
        curve_cls=BinaryCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 1},
    ),
    Config(
        mode='Race',
        data=DataSource('cml_test_race', sd_access.connect),
        meta=(
            ('W', 'White Race', 0),
            ('U', 'Unknown Race', 0),
            ('N', 'Native American Race', 0),
            ('B', 'Black Race', 0),
            ('M', 'Multiple Race', 0),
            ('A', 'Asian Race', 0),
            ('D', 'Declined Race', 0),
            ('I', 'Pacific Islander Race', 0),
            ('H', 'Hispanic Race', 0)
        ),
        curve_cls=ConstantCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 1},
    ),
    Config(
        mode='Sex',
        data=DataSource('cml_test_sex', sd_access.connect),
        meta=(
            ('M', 'Male Sex', 0),
            ('F', 'Female Sex', 0),
            ('U', 'Unknown Sex', 0)
        ),
        curve_cls=ConstantCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 1},
    ),
    Config(
        mode='Age',
        data=DataSource('cml_test_age', sd_access.connect),
        meta=(('age', 'Age', 0), ),
        curve_cls=AgeCurveBuilder,
        std_cls=LogGelmanStandardizerWithFallbacks,
        std_kws={'eps': 1e-6},
    ),
    Config(
        mode='Bmi',
        data=DataSource('cml_test_bmi', sd_access.connect),
        meta=(('BMI', 'Clean BMI', 26.12), ),
        curve_cls=RegressionCurveBuilder,
        std_cls=GelmanStandardizer,
        std_kws={'log_transform': True, 'eps': 0},
    ),
    Config(
        mode='ANA',
        data=DataSource('cml_test_ana', sd_access.connect),
        meta=(('ANA titer', 'ANA titer', 8.0), ),
        curve_cls=RegressionCurveBuilder,
        std_cls=GelmanStandardizer,
        std_kws={'log_transform': True, 'eps': 0},
    ),
]

if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')

    cache = PickleCache(loc='/hd1/stilljm/cml_tests/B')
    experiment = e = Experiment(configs, cache)
    experiment.fetch_data()
    experiment.fetch_meta()
    experiment.compute_curves(max_workers=48)
    experiment.make_standardizer()

    for i in range(10):
        path = Path(cache.loc/f'segment_{i:03}')
        with cache.relocate(path):
            experiment.compute_cross_sections(curves_key='../curves')
            experiment.build_data_matrix(meta_key='../meta')
            experiment.standardize_data_matrix(std_key='../standardizer',
                                               meta_key='../meta')
            experiment.learn_model()
            experiment.plot_model(pdf_path=cache.loc/'phenotypes.pdf',
                                  meta_key='../meta',
                                  std_key='../standardizer')

    #experiment.combine_models()
    #experiment.plot_model()
