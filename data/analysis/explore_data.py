"""
Some basic exploratory data analysis

Usage:
    $ poetry run python -m data.analysis.explore_data \
            --proportion_random_sample 0.01
"""

import argparse
import logging
import random
from typing import Final

import pandas as pd

INPUT_DATA_PATH: Final[str] = "data/input/kaggle_alaxi_paysim1.csv"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "-p",
    "--proportion_random_sample",
    help="Proportion of full dataset to use for analysis",
    type=float,
    required=True,
)
args = arg_parser.parse_args()

logger.info("Reading in input data")
data = pd.read_csv(
    INPUT_DATA_PATH,
    header=0,
    skiprows=lambda i: i > 0 and random.random() > args.proportion_random_sample,
)
logger.info("Random sample has %s rows", f"{data.shape[0]:,}")

