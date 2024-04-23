"""
Script which takes the raw input data and converts it into a
form suitable for training a model

Usage:
    $ poetry run python -m feature_eng.create_train_data
"""

import logging
from typing import Final

import duckdb

import feature_eng

INPUT_DATA_FILEPATH: Final[str] = "data/input/simdata.csv"
OUTPUT_DATA_FILEPATH: Final[str] = "data/output/train_data.csv"

# set up python logger #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

logger.info("reading input data from %s", INPUT_DATA_FILEPATH)

with duckdb.connect() as con:
    con.execute("CREATE SCHEMA fraud;")
    con.execute(
        """
    CREATE TABLE fraud.train_data (
            tid UINTEGER PRIMARY KEY
        ,   src VARCHAR(8)
        ,   dst VARCHAR(8)
        ,   amt DECIMAL(6,2)
        ,   time UTINYINT
        ,   day_of_week UTINYINT
        ,   is_fraud BOOLEAN
    )
    ;
    """
    )
    con.execute(
        f"""
    INSERT INTO fraud.train_data
    SELECT *
    FROM   read_csv('{INPUT_DATA_FILEPATH}')
    ;
    """,
    )
    con.execute(
        f"""
    CREATE TABLE fraud.aug_train_data
    AS 
    SELECT      tn.*
            ,   nt.n_transactions
            ,   dw.n_transactions_this_day_of_week
            ,   dw.n_transactions_this_day_of_week / nt.n_transactions AS prop_transactions_this_day_of_week
    FROM        fraud.train_data tn
    LEFT JOIN   (
                SELECT      src
                        ,   COUNT(*) AS n_transactions
                FROM        fraud.train_data
                GROUP BY    src
                ) nt
            ON  tn.src = nt.src  
    LEFT JOIN   (
                SELECT      src
                        ,   day_of_week
                        ,   COUNT(*) AS n_transactions_this_day_of_week
                FROM        fraud.train_data
                GROUP BY    src
                        ,   day_of_week
                ) dw
            ON  tn.src = dw.src
            AND tn.day_of_week = dw.day_of_week
    ;
    """
    )
    rel = con.sql(
        """
    SELECT  * 
    FROM    fraud.aug_train_data 
    WHERE   src = (SELECT src FROM fraud.aug_train_data USING SAMPLE 1)
    ;
            """
    )
    rel.show()
    # for row in con.fetchall():
    #     print(row)

# print(feature_eng.SQL_QUERY_PERSON_CUM_PROP_TRANSACTIONS_THIS_TIME_OF_DAY)
