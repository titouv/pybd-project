import pandas as pd
import numpy as np
import sklearn
import glob
import time
import re
import timescaledb_model as tsdb
from timescaledb_model import initial_markets_data
import dateutil
import mylogging  # Import the logging library

TSDB = tsdb.TimescaleStockMarketModel
HOME = "/home/bourse/data/"   # we expect subdirectories boursorama and euronext
HOME = "./data/" # for local testing
BATCH_SIZE = 100_000  # Default batch size for database writes

# Instantiate logger
logger = mylogging.getLogger(__name__, filename="/tmp/etl.log")

#=================================================
# Extract, Transform and Load data in the database
#=================================================

def is_data_present():
    import os
    absolute_bourso_path = os.path.abspath(HOME + 'bourso')
    absolute_euronext_path = os.path.abspath(HOME + 'euronext')
    return os.path.exists(absolute_bourso_path) and os.path.exists(absolute_euronext_path)


def verify_db_state(db:TSDB):
    # clear table companies, stocks and daystocks
    logger.info("Clearing database tables: companies, stocks, daystocks")
    db._purge_database()
    # db._setup_database()

    
# private functions
def read_raw_bousorama(years:list[str]):
    dfs = []
    for year in years:
        compA = pd.concat({dateutil.parser.parse(f.split('compA ')[1].split('.bz2')[0]):pd.read_pickle(f) for f in glob.glob(HOME + 'bourso/' + year + '/compA*')})
        compB = pd.concat({dateutil.parser.parse(f.split('compB ')[1].split('.bz2')[0]):pd.read_pickle(f) for f in glob.glob(HOME + 'bourso/' + year + '/compB*')})
        merge = pd.concat([compA, compB])
        dfs.append(merge)
    return pd.concat(dfs)

def clean_raw_bousorama(df):
    return df

def make_subset_of_companies_bousorama(df):
    # make a copy of whole datframe that contains only the unique combination of symbol + name and only those columns
    df_unique = df[['symbol', 'name']].drop_duplicates(subset=['symbol', 'name']).reset_index(drop=True)
    return df_unique

def add_market_column_boursorama(df):
    # add a column 'Market' to the df_unique and cleaned symbol
    initial_markets_data = (
        (1, "New York", "nyse", "", "NYSE", ""),
        (2, "London Stock Exchange", "lse", "1u*.L", "LSE", ""),
        (3, "Bourse de Milan", "milano", "1g", "", ""),
        (4, "Mercados Espanoles", "mercados", "FF55-", "", ""),
        (5, "Amsterdam", "amsterdam", "1rA", "", "Amsterdam"),
        (6, "Paris", "paris", "1rP", "ENXTPA", "Paris"),
        (7, "Deutsche Borse", "xetra", "1z", "", ""),
        (8, "Bruxelle", "bruxelle", "FF11_", "", "Brussels"),
        (9, "Australie", "asx", "", "ASX", ""),
        (100, "International", "int", "", "", ""),  # should be last one
    )
    df['market'] = df['symbol'].apply(lambda x: next(
        (market[1] for market in initial_markets_data 
         if market[3] and market[3] in x),
        "International"  # Default value if no match found
    ))
    
    # Add cleaned symbol column by removing market prefixes
    df['cleaned_symbol'] = df['symbol'].apply(lambda x: next(
        (x.replace(market[3], '') for market in initial_markets_data 
         if market[3] and market[3] in x),
        x  # Keep original if no prefix found
    ))
    return df


def make_normalized_dataframe_boursorama(df):
    # Columns should be : symbol, name, market and boursorama (for the raw value)
    # rename cleaned_symbol to symbol
    df.rename(columns={'symbol':'boursorama'}, inplace=True)
    df.rename(columns={'cleaned_symbol': 'symbol'}, inplace=True)
    return df



def read_raw_euronext(years:list[str]):
    # raw columns name are
    # Name,ISIN,Symbol,	Market,	Trading, Currency, Open, High, Low, Last, Last Date/Time, Time Zone, Volume, Turnover
    def read_euronext_file(path):
        if path.endswith(".csv"):
            return pd.read_csv(path, delimiter='\t')
        return pd.read_excel(path)

    dfs = []
    for year in years:
        files = glob.glob(HOME + 'euronext/*' + year + '*')
        df =  pd.concat([read_euronext_file(f) for f in files])
        dfs.append(df)
    return pd.concat(dfs)

def clean_raw_euronext(df):
    df = df.iloc[3:]
    df = df[df['Symbol'].notna()]
    return df

def make_subset_of_companies_euronext(df):
    # make a copy of whole datframe that contains only the unique combination of Name	ISIN	Symbol	Market and only those columns
    df_unique = df[['Name', 'ISIN', 'Symbol', 'Market']].drop_duplicates(subset=[
        'Name',
        'ISIN', 
        'Symbol',
        'Market'
    ])
    return df_unique.copy()

def add_market_column_euronext(df):
    # add a column 'Market' to the df_unique
    euronext_market_to_db_market = {
        "Euronext Growth Paris":"Paris",
        "Euronext Paris":"Paris",
        "Euronext Access Paris":"Paris",
        "Euronext Paris, Amsterdam":"Paris",
        "Euronext Paris, Brussels":"Paris",
        "Euronext Amsterdam, Brussels, Paris":"Paris",
        "Euronext Amsterdam, Paris":"Paris",
        "Euronext Brussels, Paris":"Paris",
        "Euronext Growth Paris, Brussels":"Paris",
        "Euronext Paris, Amsterdam, Brussels":"Paris",
        "Euronext Growth Brussels, Paris":"Paris",
        "Euronext Brussels, Amsterdam, Paris":"Paris",
        "Euronext Paris, London":"Paris",
        "Euronext Growth Dublin":"Dublin",
        "Euronext Dublin":"Dublin",
    }
    df['market'] = df['Market'].map(euronext_market_to_db_market)
    return df


def make_normalized_dataframe_euronext(df):
    # Columns should be : symbol, name, market and isin
    df.rename(columns={'Symbol': 'symbol'}, inplace=True)
    df.rename(columns={'ISIN': 'isin'}, inplace=True)
    df.rename(columns={'Name': 'name'}, inplace=True)
    # add column euronext with the raw name
    df['euronext'] = df['name']
    return df[['symbol', 'name', 'market', 'isin', 'euronext']]

#
# decorator
# 

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} run in {(end_time - start_time):.2f} seconds.")
        return result

    return wrapper

#
# public functions
# 

# Add new batch write function
def batch_df_write(df, table_name, db, batch_size=BATCH_SIZE):
    """
    Write dataframe to database in batches to reduce memory pressure.
    
    Args:
        df: DataFrame to write
        table_name: Target table name
        db: Database connection
        batch_size: Size of batches to write
    """
    total_rows = len(df)
    num_batches = (total_rows + batch_size - 1) // batch_size  # Ceiling division
    
    logger.info(f"Writing {total_rows} rows to {table_name} in {num_batches} batches")
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]
        
        logger.info(f"Writing batch {i+1}/{num_batches} ({len(batch_df)} rows) to {table_name}")
        db.df_write(batch_df, table_name)
        
    logger.info(f"Completed writing {total_rows} rows to {table_name}")

@timer_decorator
def store_files(years:list[str], db:TSDB):
    logger.info(f"Starting ETL process for years {years}")

    if not is_data_present():
        logger.error("Data not present")
        raise ValueError("Data not present")

    verify_db_state(db)

    logger.info(f"Reading raw Boursorama data for years {years}")
    raw_boursorama = read_raw_bousorama(years)
    logger.info(f"Cleaning raw Boursorama data ({len(raw_boursorama)} rows)")
    raw_boursorama = clean_raw_bousorama(raw_boursorama)
    logger.info("Extracting unique companies from Boursorama data")
    companies_bousorama = make_subset_of_companies_bousorama(raw_boursorama)
    logger.info(f"Adding market information to Boursorama companies ({len(companies_bousorama)} rows)")
    companies_bousorama = add_market_column_boursorama(companies_bousorama)
    logger.info("Normalizing Boursorama companies dataframe")
    companies_bousorama = make_normalized_dataframe_boursorama(companies_bousorama)

    logger.info(f"Reading raw Euronext data for years {years}")
    raw_euronext = read_raw_euronext(years)
    logger.info(f"Cleaning raw Euronext data ({len(raw_euronext)} rows)")
    raw_euronext = clean_raw_euronext(raw_euronext)
    logger.info("Extracting unique companies from Euronext data")
    companies_euronext = make_subset_of_companies_euronext(raw_euronext)
    logger.info(f"Adding market information to Euronext companies ({len(companies_euronext)} rows)")
    companies_euronext = add_market_column_euronext(companies_euronext)
    logger.info("Normalizing Euronext companies dataframe")
    companies_euronext = make_normalized_dataframe_euronext(companies_euronext)

    logger.info("Merging Boursorama and Euronext company data")
    merged_companies = pd.merge(companies_euronext, companies_bousorama, 
                            left_on=['symbol', 'name', 'market'], 
                            right_on=['symbol', 'name', 'market'], 
                            how='outer')
    print(merged_companies.head())
    logger.info(f"Merged companies dataframe created ({len(merged_companies)} rows)")
    
    companies_db_dataframe = pd.DataFrame(
        columns=[
            "id", "name", "mid", "symbol", "isin", "boursorama", "euronext", 
            # not used, ricou explains why in moodle question
            # "pea", "sector1", "sector2", "sector3"
        ],
    )

    # generate id
    companies_db_dataframe['id'] = merged_companies.index
    companies_db_dataframe['name'] = merged_companies['name']
    # adapt from name (2 columns) to mid (1 column)
    companies_db_dataframe['mid'] = merged_companies['market'].apply(lambda x: next(
        (market[1] for market in initial_markets_data 
         if market[3] and market[3] in x),
        "100"  # Default value if no match found
    ))
    companies_db_dataframe['symbol'] = merged_companies['symbol']
    companies_db_dataframe['isin'] = merged_companies['isin']
    # TODO see to fill column boursorama and euronext
    companies_db_dataframe['boursorama'] = merged_companies['boursorama']
    companies_db_dataframe['euronext'] = merged_companies['euronext']

    print(companies_db_dataframe.head())

    logger.info("Writing companies data to database")
    batch_df_write(companies_db_dataframe, 'companies', db)
    logger.info(f"Stored {len(companies_db_dataframe)} companies")

    # add column company_id into raw_boursorama and raw_euronext
    bousorama_to_id = dict(zip(companies_db_dataframe['boursorama'], companies_db_dataframe['id']))
    raw_boursorama['company_id'] = raw_boursorama['symbol'].map(bousorama_to_id)

    euronext_to_id = dict(zip(companies_db_dataframe['euronext'], companies_db_dataframe['id']))
    raw_euronext['company_id'] = raw_euronext['Name'].map(euronext_to_id)
    
    logger.info("Preparing daystocks data from Euronext")
    # create daystocks for euronext
    daystocks_db_dataframe = pd.DataFrame(
        columns=[
           "date", "cid", "open", "close", "high", "low", "volume", "mean", "std"
        ],
    )
    daystocks_db_dataframe['date'] = pd.to_datetime(raw_euronext['Last Date/Time'], 
                                                   format='%d/%m/%y %H:%M',
                                                   errors='coerce')
    daystocks_db_dataframe['cid'] = raw_euronext['company_id']
    
    # Convert columns to numeric, replacing invalid values with NaN
    daystocks_db_dataframe['open'] = pd.to_numeric(raw_euronext['Open'], errors='coerce')
    daystocks_db_dataframe['close'] = pd.to_numeric(raw_euronext['Last'], errors='coerce')
    daystocks_db_dataframe['high'] = pd.to_numeric(raw_euronext['High'], errors='coerce')
    daystocks_db_dataframe['low'] = pd.to_numeric(raw_euronext['Low'], errors='coerce')
    daystocks_db_dataframe['volume'] = pd.to_numeric(raw_euronext['Volume'], errors='coerce')
    #remove line where date is nan
    daystocks_db_dataframe = daystocks_db_dataframe[daystocks_db_dataframe['date'].notna()]
    # TODO compute mean and std
    

    logger.info("Writing daystocks data to database")
    batch_df_write(daystocks_db_dataframe, 'daystocks', db)
    logger.info(f"Stored {len(daystocks_db_dataframe)} daystocks entries")

    logger.info("Preparing stocks data from Boursorama")
    # create stocks for boursorama
    stocks_db_dataframe = pd.DataFrame(
        columns=[
           "date", "cid", "value", "volume"
        ],
    )

    # default index for raw_boursorama is a "date" and "symbol"
    # we want to convert this index to a "date" only
    raw_boursorama = raw_boursorama.drop(columns=['symbol'])
    raw_boursorama.reset_index(inplace=True)

    raw_boursorama['date'] = pd.to_datetime(raw_boursorama['level_0'])

    stocks_db_dataframe['date'] =  raw_boursorama['date']

    def get_clean_last_boursorama(df):
        """last is of object type and sometimes ends with (c) or (s)"""
        return [
            float(re.split("\\(.\\)$", str(x))[0].replace(" ", "").replace(",", "."))
            for x in df["last"]
        ]

    stocks_db_dataframe['cid'] = raw_boursorama['company_id']
    stocks_db_dataframe['value'] = get_clean_last_boursorama(raw_boursorama)
    stocks_db_dataframe['volume'] = raw_boursorama['volume']
    stocks_db_dataframe

    logger.info("Writing stocks data to database")
    batch_df_write(stocks_db_dataframe, 'stocks', db)
    logger.info(f"Stored {len(stocks_db_dataframe)} stocks entries")

    logger.info("ETL process finished successfully.")
    
    
    

     




if __name__ == '__main__':
    print("Go Extract Transform and Load")
    pd.set_option('display.max_columns', None)  # usefull for dedugging
    # db = tsdb.TimescaleStockMarketModel('bourse', 'ricou', 'db', 'monmdp')        # inside docker
    db = tsdb.TimescaleStockMarketModel('bourse', 'ricou', 'localhost', 'monmdp') # outside docker
    years = ["2020", "2021", "2022", "2023", "2024"]
    store_files(years, db)
    print("Done Extract Transform and Load")
