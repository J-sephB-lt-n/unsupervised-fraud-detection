"""
Script which takes the raw input data and converts it into a
form suitable for training a model

Usage:
    $ poetry run python -m feature_eng.create_train_data
"""

import logging
from typing import Final

import duckdb

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
    USING  SAMPLE 100 PERCENT
    ;
    """,
    )
    con.execute(
        f"""
    CREATE TABLE fraud.aug_train_data
    AS 
    SELECT      trn.*
            --,   src.n_transactions
            --,   src.min_amt
            --,   src.mean_amt
            --,   src.median_amt
            --,   src.max_amt
            ,   amt / src.min_amt AS src_ratio_to_min_amt
            ,   amt / src.mean_amt AS src_ratio_to_mean_amt
            ,   amt / src.median_amt AS src_ratio_to_median_amt
            ,   amt / src.max_amt AS src_ratio_to_max_amt
            --,   sdw.src_n_transactions_this_day_of_week
            ,   sdw.src_n_transactions_this_day_of_week / src.n_transactions AS prop_transactions_this_day_of_week
            --,   stm.src_n_transactions_this_time
            ,   stm.src_n_transactions_this_time / src.n_transactions AS prop_transactions_this_time
    FROM        fraud.train_data trn
    LEFT JOIN   (
                SELECT      src
                        ,   COUNT(*) AS n_transactions
                        ,   MIN(amt) AS min_amt
                        ,   MEDIAN(amt) AS median_amt
                        ,   AVG(amt) as mean_amt
                        ,   MAX(amt) AS max_amt
                FROM        fraud.train_data
                GROUP BY    src
                ) src
            ON  trn.src = src.src  
    LEFT JOIN   (
                SELECT      src
                        ,   day_of_week
                        ,   COUNT(*) AS src_n_transactions_this_day_of_week
                FROM        fraud.train_data
                GROUP BY    src
                        ,   day_of_week
                ) sdw
            ON  trn.src = sdw.src
            AND trn.day_of_week = sdw.day_of_week
    LEFT JOIN   (
                SELECT      src
                        ,   time
                        ,   COUNT(*) AS src_n_transactions_this_time
                FROM        fraud.train_data
                GROUP BY    src
                        ,   time
                ) stm
            ON  trn.src = stm.src
            AND trn.time = stm.time
    ;
    """
    )
    logger.info("Completed dataset creation - starting export to %s", OUTPUT_DATA_FILEPATH)
    con.execute(f"COPY fraud.aug_train_data TO '{OUTPUT_DATA_FILEPATH}' (HEADER, DELIMITER ',');")
    logger.info("Finished exporting data to %s", OUTPUT_DATA_FILEPATH)
    print("Here is a random sample of the exported data:")
    rel = con.sql(
        """
    SELECT  * 
    FROM    fraud.aug_train_data 
    WHERE   src = (SELECT src FROM fraud.aug_train_data USING SAMPLE 1)
    ;
            """
    )
    rel.show()
