"""
SQL script to generate calculated fields related to the person-wise 
proportion of transactions which are at a particular time of day
"""

from typing import Final

PERSON_PROPORTION_TRANSACTIONS_THIS_TIME_OF_DAY: Final[str] = """
ALTER TABLE fraud.train_data
ADD COLUMN 
"""
