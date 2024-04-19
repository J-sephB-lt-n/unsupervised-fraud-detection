"""
Script which takes the raw input data and converts it into a
form suitable for training a model

Usage:
    $ poetry run python -m models.feature_eng.create_train_data
"""

import logging
from typing import Final

import pandas as pd

INPUT_DATA_FILEPATH: Final[str] = "data/input/kaggle_alaxi_paysim1.csv"
OUTPUT_DATA_FILEPATH: Final[str] = "data/output/train_data.csv"

# set up python logger #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.info("reading input data from %s", INPUT_DATA_FILEPATH)
data = pd.read_csv(INPUT_DATA_FILEPATH, nrows=10_000)

print(data)
