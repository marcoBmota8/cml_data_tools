# %% 
from curves import CategoricalCurveBuilder, build_patient_curves
import pandas as pd
import numpy as np
# %%
# Generate object sfor toy example
spec = {
    'Categorical labs': CategoricalCurveBuilder()
}

# Generate random data
all_dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
data_dates = np.random.choice(all_dates, size=10, replace=False)
data = pd.DataFrame(
    {
    'date': data_dates,
    'ptid': 10*['Rtest'],
    'mode': 10*['Categorical labs'],
    'channel': 10*['ANA_pattern'],
    'value': np.random.choice(['smooth', 'speckle', 'nucleolar'], size = 10)
    })

# Generate grid of timestamps
grid = pd.date_range(start='2023-01-01', periods=20, freq='5D')


# %%
build_patient_curves(
    df = data,
    spec=spec
)
    # %%
