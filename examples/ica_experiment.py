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
    loc = '/hd1/stilljm/cml_tests/C'

    logging.basicConfig(filename='ICA_run_C.log',
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    logging.info(f'Setting up Experiment with cache loc {loc}')

    cache = PickleCache(loc=loc)
    experiment = e = Experiment(configs, cache)

    logging.info('Fetching data & channel metadata')
    experiment.fetch_data()
    experiment.fetch_meta()

    logging.info('Computing Curves')
    experiment.compute_curves(max_workers=48)

    logging.info('Constructing standardizer')
    experiment.make_standardizer()

    logging.info('Beginning to train submodels')
    for i in range(100):
        logging.info('Training submodel {i:03}')
        path = Path(cache.loc/f'segment_{i:03}')
        with cache.relocate(path):
            experiment.compute_cross_sections(curves_key='../curves')
            # create and standardize data matrix in one step
            experiment.build_standardized_data_matrix(key='std_matrix',
                                                      meta_key='../meta',
                                                      std_key='../standardizer',
                                                      save_dense=False)
            experiment.learn_model()
            # Remove the std matrix (~40G) for storage considerations
            cache.remove('std_matrix')

    logging.info('Submodel training complete')
    model_keys = sorted(cache.loc.glob('segment_*/model.pkl'))

#    # Generate PDFs if they don't already exist
#    pdf_dir = cache.loc/'pdfs'
#    try:
#        pdf_dir.mkdir()
#    except FileExistsError:
#        pass
#    else:
#        for i, model_key in enumerate(model_keys):
#            experiment.plot_model(pdf_path=pdf_dir/f'phenotypes_{i:03}.pdf',
#                                  model_key=model_key)

    # Collect phenotypes
    logging.info('Collecting submodel phenotypes together')
    experiment.collect_phenotypes(model_keys)
