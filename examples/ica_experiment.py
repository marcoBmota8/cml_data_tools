#!/usr/bin/env python
"""
Example application using ICA
"""
from pathlib import Path

import sd_access
from cml_data_tools.source_ehr import DataSource, MetaSource
from cml_data_tools.config import Config
from cml_data_tools.experiment import Experiment
from cml_data_tools.curves import *
from cml_data_tools.standardizers import *


configs = [
    Config(
        mode='lab_value',
        data=DataSource('cml_20_lab_values', sd_access.connect),
        meta=MetaSource('cml_20_lab_values_meta', sd_access.connect),
        curve_cls=RegressionCurveBuilder,
        std_cls=YeoJohnsonStandardizer,
    ),
    Config(
        mode='phecode',
        data=DataSource('cml_20_phecodes', sd_access.connect),
        meta=MetaSource('cml_20_phecodes_meta', sd_access.connect),
        curve_cls=IntensityCurveBuilder,
        std_cls=LogStandardizer,
        std_kws={'pre_scale_factor': 20 * 365.25},
    ),
    Config(
        mode='medication',
        data=DataSource('cml_20_meds', sd_access.connect),
        meta=MetaSource('cml_20_meds_meta', sd_access.connect),
        curve_cls=BinaryCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 2},
    ),
    Config(
        mode='race',
        data=DataSource('cml_20_race', sd_access.connect),
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
        std_kws={'scale': 4},
    ),
    Config(
        mode='sex',
        data=DataSource('cml_20_gender', sd_access.connect),
        meta=(
            ('M', 'Male Sex', 0),
            ('F', 'Female Sex', 0),
            ('U', 'Unknown Sex', 0)
        ),
        curve_cls=ConstantCurveBuilder,
        std_cls=LinearStandardizer,
        std_kws={'scale': 4},
    ),
    Config(
        mode='age',
        data=DataSource('cml_20_age', sd_access.connect),
        meta=(('age', 'Age', 0), ),
        curve_cls=AgeCurveBuilder,
        std_cls=SquaredStandardizer,
        std_kws={'scale': 8 / (80**2)},
    ),
    Config(
        mode='surgical_cpt',
        data=DataSource('cml_20_cpt', sd_access.connect),
        meta=MetaSource('cml_20_cpt_meta', sd_access.connect),
        curve_cls=LogCumulativeCurveBuilder,
        std_cls=LogStandardizer,
        std_kws={'pre_scale_factor': 1},
    ),
]

if __name__ == '__main__':
    experiment = e = Experiment(configs, loc='cache')
    experiment.fetch_data()
    experiment.fetch_meta()
    experiment.compute_curves()
    experiment.compute_cross_sections()
    experiment.make_standardizer()
    experiment.build_data_matrix()
    experiment.standardize_data_matrix()
    experiment.learn_model(max_phenotypes=10, max_iter=100)
