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
#HOME = "./data/" # for local testing
BATCH_SIZE = 100_000  # Default batch size for database writes

# Instantiate logger
logger = mylogging.getLogger(__name__)

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
def read_raw_bousorama(year:str):
    compA_files = glob.glob(HOME + 'bourso/' + year + '/compA*')
    compB_files = glob.glob(HOME + 'bourso/' + year + '/compB*')
    compA = pd.concat({dateutil.parser.parse(f.split('compA ')[1].split('.bz2')[0]):pd.read_pickle(f) for f in compA_files})    
    compB = pd.concat({dateutil.parser.parse(f.split('compB ')[1].split('.bz2')[0]):pd.read_pickle(f) for f in compB_files})
    merge = pd.concat([compA, compB])
    return merge

def clean_raw_bousorama(df):
    df['name'] = df['name'].str.replace(r'^SRD\s+', '', regex=True)
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



def read_raw_euronext(year:str):
    # raw columns name are
    # When CSV: Name, ISIN, Symbol, Market, Trading, Currency, Open, High, Low, Last, Last Date/Time, Time Zone, Volume, Turnover
    # When Excel: Name, ISIN, Symbol, Market, Currency, Open Price, High Price, low Price, last Price, last Trade MIC Time, Time Zone, Volume, Turnover, European Equities	
    def read_euronext_file(path):
        if path.endswith(".csv"):
            return pd.read_csv(path, delimiter='\t')
        # when reading excel remap to csv column name
        return pd.read_excel(path).rename(columns={
            "Open Price":"Open",
            "High Price":"High",
            "low Price":"Low",
            "last Price":"Last",
            "last Trade MIC Time":"Last Date/Time"
        })

    files = glob.glob(HOME + 'euronext/*' + year + '*')
    df =  pd.concat([read_euronext_file(f) for f in files])
    return df

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

def get_clean_last_boursorama(df):
    """last is of object type and sometimes ends with (c) or (s)"""
    return [
        float(re.split("\\(.\\)$", str(x))[0].replace(" ", "").replace(",", "."))
        for x in df["last"]
    ]

def handle_stocks(raw_boursorama, db):
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


    stocks_db_dataframe['cid'] = raw_boursorama['company_id']
    stocks_db_dataframe['value'] = get_clean_last_boursorama(raw_boursorama)
    stocks_db_dataframe['volume'] = raw_boursorama['volume']
    
    # Remove rows with missing company IDs as they cannot be processed correctly
    initial_rows = len(stocks_db_dataframe)
    stocks_db_dataframe.dropna(subset=['cid'], inplace=True)
    if initial_rows > len(stocks_db_dataframe):
        logger.warning(f"Removed {initial_rows - len(stocks_db_dataframe)} rows with missing company IDs (cid).")

    # Ensure cid is integer type for proper grouping
    stocks_db_dataframe['cid'] = stocks_db_dataframe['cid'].astype(int)

    # --- Filtering logic starts here ---
    logger.info(f"Initial number of stock entries: {len(stocks_db_dataframe)}")

    # 1. Sort by company ID and date
    stocks_db_dataframe.sort_values(by=['cid', 'date'], inplace=True)

    # 2. Remove rows with volume == 0
    rows_before_volume_filter = len(stocks_db_dataframe)
    stocks_db_dataframe = stocks_db_dataframe[stocks_db_dataframe['volume'] != 0]
    rows_after_volume_filter = len(stocks_db_dataframe)
    logger.info(f"Removed {rows_before_volume_filter - rows_after_volume_filter} rows with volume = 0")

    # 3. Identify rows where value is unchanged compared to the previous and next row for the same company
    stocks_db_dataframe['prev_value'] = stocks_db_dataframe.groupby('cid')['value'].shift(1)
    stocks_db_dataframe['next_value'] = stocks_db_dataframe.groupby('cid')['value'].shift(-1)

    # Keep a row if:
    # - It's the first or last record for a company (prev_value or next_value is NaN)
    # - The value is different from the previous value
    # - The value is different from the next value
    keep_mask = (
        stocks_db_dataframe['prev_value'].isna() | 
        stocks_db_dataframe['next_value'].isna() | 
        (stocks_db_dataframe['value'] != stocks_db_dataframe['prev_value']) |
        (stocks_db_dataframe['value'] != stocks_db_dataframe['next_value'])
    )
    
    filtered_stocks_df = stocks_db_dataframe[keep_mask]
    
    # Drop helper columns
    filtered_stocks_df = filtered_stocks_df.drop(columns=['prev_value', 'next_value'])
    
    rows_after_value_filter = len(filtered_stocks_df)
    logger.info(f"Removed {rows_after_volume_filter - rows_after_value_filter} consecutive rows with unchanged value")
    logger.info(f"Final number of stock entries after filtering: {rows_after_value_filter}")
    # --- Filtering logic ends here ---

    logger.info("Writing filtered stocks data to database")
    # Use the filtered dataframe for writing
    batch_df_write(filtered_stocks_df, 'stocks', db)
    logger.info(f"Stored {len(filtered_stocks_df)} filtered stocks entries")


def handle_daystocks(raw_euronext,raw_boursorama, db):
    logger.info("Preparing daystocks data from Euronext")

       # create daystocks for euronext
    daystocks_db_dataframe = pd.DataFrame(
        columns=[
           "date", "cid", "open", "close", "high", "low", "volume", "mean", "std"
        ],
    )

    def parse_date(date_str):
        try:
            # First try with 4-digit year format
            return pd.to_datetime(date_str, format='%d/%m/%Y %H:%M')
        except:
            try:
                # Then try with 2-digit year format
                return pd.to_datetime(date_str, format='%d/%m/%y %H:%M')
            except:
                return pd.NaT

    daystocks_db_dataframe['date'] = raw_euronext['Last Date/Time'].apply(parse_date)
    daystocks_db_dataframe['cid'] = raw_euronext['company_id']
    
    # Convert columns to numeric, replacing invalid values with NaN
    daystocks_db_dataframe['open'] = pd.to_numeric(raw_euronext['Open'], errors='coerce')
    daystocks_db_dataframe['close'] = pd.to_numeric(raw_euronext['Last'], errors='coerce')
    daystocks_db_dataframe['high'] = pd.to_numeric(raw_euronext['High'], errors='coerce')
    daystocks_db_dataframe['low'] = pd.to_numeric(raw_euronext['Low'], errors='coerce')
    daystocks_db_dataframe['volume'] = pd.to_numeric(raw_euronext['Volume'], errors='coerce')
    print("daystocks_db_dataframe", daystocks_db_dataframe)
    #remove line where date is nan
    print('remove nan')
    daystocks_db_dataframe = daystocks_db_dataframe[daystocks_db_dataframe['date'].notna()]
    # dedupe lines with the same date and cid
    print(f"Number of rows before deduplication: {len(daystocks_db_dataframe)}")
    daystocks_db_dataframe = daystocks_db_dataframe.drop_duplicates(subset=['date', 'cid'])
    print(f"Number of rows after deduplication: {len(daystocks_db_dataframe)}")


    # TODO compute mean and std

    print("daystocks_db_dataframe", daystocks_db_dataframe)

    # compute and insert daystocks for boursorama
    logger.info("Aggregating Boursorama data for daily stats")
    
    # Ensure 'date' is the index for efficient resampling and sorting
    # Make a copy to avoid modifying the original raw_boursorama needed for stocks
    raw_boursorama_agg = raw_boursorama.copy()
    # Reset index to bring 'level_0' (date) into columns, keeping the inner index
    raw_boursorama_agg.reset_index(level=0, inplace=True) 
    # Rename 'level_0' to 'date'
    raw_boursorama_agg.rename(columns={'level_0': 'date'}, inplace=True) 
    # Ensure the new 'date' column is datetime type
    raw_boursorama_agg['date'] = pd.to_datetime(raw_boursorama_agg['date']) 
    # Now set the 'date' column as the index
    raw_boursorama_agg = raw_boursorama_agg.set_index('date').sort_index() 

 

    # Clean the 'last' column *before* aggregation
    raw_boursorama_agg['value'] = get_clean_last_boursorama(raw_boursorama_agg)



    print("raw_boursorama_agg", raw_boursorama_agg.head())
    # Aggregate daily data
    daystocks_boursorama = raw_boursorama_agg.groupby('company_id').resample('D').agg(
        open=('value', 'first'),
        high=('value', 'max'),
        low=('value', 'min'),
        close=('value', 'last'),
        volume=('volume', 'sum'),
        mean=('value', 'mean'),
        std=('value', 'std')
    ).reset_index() # Reset index to get 'company_id' and 'date' back as columns

    # Rename columns to match daystocks_db_dataframe
    daystocks_boursorama.rename(columns={'company_id': 'cid'}, inplace=True)

    print("daystocks_boursorama", daystocks_boursorama.head())
    
    # Keep only relevant columns and ensure correct order
    daystocks_boursorama = daystocks_boursorama[['date', 'cid', 'open', 'close', 'high', 'low', 'volume', 'mean', 'std']]
    

    print("Number of rows before dropping nan", len(daystocks_boursorama))
    # Debug: Check for NaN values in each column
    print("NaN counts in each column:")
    print(daystocks_boursorama[['open', 'close', 'high', 'low', 'volume']].isna().sum())
    # Remove rows with NaN values resulting from aggregation (e.g., days with no trades)
    daystocks_boursorama.dropna(subset=['open', 'close', 'high', 'low', 'volume'], how='any', inplace=True)
    print("Number of rows after dropping nan", len(daystocks_boursorama))
    
    logger.info(f"Aggregated {len(daystocks_boursorama)} daily entries from Boursorama")

    # Combine Euronext and Boursorama daystocks
    logger.info("Combining Euronext and Boursorama daystocks data")

    # Identify companies already present in Euronext data (daystocks_db_dataframe)
    euronext_cids = daystocks_db_dataframe['cid'].unique()
    logger.info(f"Found {len(euronext_cids)} unique company IDs in Euronext data.")
    print("daystocks_db_dataframe", daystocks_db_dataframe)

    # Filter Boursorama data to exclude companies already in Euronext data
    daystocks_boursorama_filtered = daystocks_boursorama[~daystocks_boursorama['cid'].isin(euronext_cids)]
    original_boursorama_count = len(daystocks_boursorama)
    filtered_boursorama_count = len(daystocks_boursorama_filtered)
    excluded_boursorama_count = original_boursorama_count - filtered_boursorama_count
    logger.info(f"Filtered Boursorama data: keeping {filtered_boursorama_count} rows (excluded {excluded_boursorama_count} rows for companies already in Euronext).")

    logger.info("Concatenating %d daystocks from Euronext and %d filtered daystocks from Boursorama", len(daystocks_db_dataframe), filtered_boursorama_count)
    # Concatenate the original Euronext data with the filtered Boursorama data
    daystocks_db_dataframe = pd.concat([daystocks_db_dataframe, daystocks_boursorama_filtered], ignore_index=True)
    daystocks_db_dataframe['date'] = pd.to_datetime(daystocks_db_dataframe['date'])

    # Sort the final dataframe
    daystocks_db_dataframe.sort_values(by=['date', 'cid'], inplace=True)

    logger.info(f"Combined dataframe now contains {len(daystocks_db_dataframe)} rows.")

    logger.info("Writing daystocks data to database")
    batch_df_write(daystocks_db_dataframe, 'daystocks', db)
    logger.info(f"Stored {len(daystocks_db_dataframe)} daystocks entries")

def handle_companies(companies_euronext, companies_bousorama,raw_boursorama,raw_euronext, db):
    global full_companies_db_dataframe
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
        ],
    )

    # generate id
    companies_db_dataframe['id'] = merged_companies.index
    companies_db_dataframe['name'] = merged_companies['name']
    # adapt from name (2 columns) to mid (1 column)
    companies_db_dataframe['mid'] = merged_companies['market'].apply(lambda x: next(
        (market[0] for market in initial_markets_data 
         if market[1] and market[1] in x),
        "100"  # Default value if no match found
    ))
    companies_db_dataframe['symbol'] = merged_companies['symbol']
    companies_db_dataframe['isin'] = merged_companies['isin']
    companies_db_dataframe['boursorama'] = merged_companies['boursorama']
    companies_db_dataframe['euronext'] = merged_companies['euronext']

    # First, ensure full_companies_db_dataframe has an 'id' column if it's empty
    if len(full_companies_db_dataframe) == 0:
        full_companies_db_dataframe = pd.DataFrame(columns=[
            "id", "name", "mid", "symbol", "isin", "boursorama", "euronext"
        ])
    
    # Identify existing companies by checking all identifying fields
    existing_companies = pd.merge(
        companies_db_dataframe,
        full_companies_db_dataframe,
        on=['symbol', 'name', 'isin', 'mid'],  # Added 'mid' to ensure market matches too
        how='left',
        indicator=True,
        suffixes=('', '_existing')
    )
    
    # For companies that exist, use their existing IDs
    mask_existing = existing_companies['_merge'] == 'both'
    companies_db_dataframe.loc[mask_existing, 'id'] = existing_companies.loc[mask_existing, 'id_existing']
    
    # Find truly new companies (those not in full_companies_db_dataframe)
    new_companies = companies_db_dataframe[~mask_existing].copy()
    
    if len(new_companies) > 0:
        # Generate new IDs for new companies
        max_id = full_companies_db_dataframe['id'].max() if len(full_companies_db_dataframe) > 0 else -1
        if pd.isna(max_id):
            max_id = -1
        new_companies['id'] = range(int(max_id) + 1, int(max_id) + 1 + len(new_companies))
        
        # Ensure id is integer type
        new_companies['id'] = new_companies['id'].astype(int)
        
        # Add new companies to full_companies_db_dataframe
        full_companies_db_dataframe = pd.concat([full_companies_db_dataframe, new_companies], ignore_index=True)
        
        # Sort by ID for consistency
        full_companies_db_dataframe.sort_values(by=['id'], inplace=True)
        full_companies_db_dataframe.reset_index(drop=True, inplace=True)
        
        # Write only the new companies to the database
        logger.info(f"Writing {len(new_companies)} new companies to database")
        batch_df_write(new_companies, 'companies', db)
        logger.info(f"Stored {len(new_companies)} new companies")
    else:
        logger.info("No new companies to write to database")
    
    print("ID range:", full_companies_db_dataframe['id'].min(), "to", full_companies_db_dataframe['id'].max())
    print(f"Total number of companies after processing : {len(full_companies_db_dataframe)}")

    # Update company_id mappings for raw data using the full_companies_db_dataframe
    # This ensures we use consistent IDs for both new and existing companies
    bousorama_to_id = dict(zip(full_companies_db_dataframe['boursorama'], full_companies_db_dataframe['id']))
    raw_boursorama['company_id'] = raw_boursorama['symbol'].map(bousorama_to_id)

    euronext_to_id = dict(zip(full_companies_db_dataframe['euronext'], full_companies_db_dataframe['id']))
    raw_euronext['company_id'] = raw_euronext['Name'].map(euronext_to_id)


def load_year(year:str, db:TSDB):
    logger.info(f"Starting ETL process for year {year}")

    logger.info(f"Reading raw Boursorama data for year {year}")
    raw_boursorama = read_raw_bousorama(year)
    logger.info(f"Cleaning raw Boursorama data ({len(raw_boursorama)} rows)")
    raw_boursorama = clean_raw_bousorama(raw_boursorama)
    logger.info("Extracting unique companies from Boursorama data")
    companies_bousorama = make_subset_of_companies_bousorama(raw_boursorama)
    logger.info(f"Adding market information to Boursorama companies ({len(companies_bousorama)} rows)")
    companies_bousorama = add_market_column_boursorama(companies_bousorama)
    logger.info("Normalizing Boursorama companies dataframe")
    companies_bousorama = make_normalized_dataframe_boursorama(companies_bousorama)

    # Check if there are any Euronext files for this year
    euronext_files = glob.glob(HOME + 'euronext/*' + year + '*')
    if not euronext_files:
        logger.warning(f"No Euronext files found for year {year}. Processing with Boursorama data only.")
        # Create empty DataFrames for Euronext data
        raw_euronext = pd.DataFrame(columns=["Name", "ISIN", "Symbol", "Market", "Trading", "Currency", "Open", "High", "Low", "Last", "Last Date/Time", "Time Zone", "Volume", "Turnover"])
        companies_euronext = pd.DataFrame(columns=['symbol', 'name', 'market', 'isin', 'euronext'])
    else:
        logger.info(f"Reading raw Euronext data for year {year}")
        raw_euronext = read_raw_euronext(year)
        logger.info(f"Cleaning raw Euronext data ({len(raw_euronext)} rows)")
        raw_euronext = clean_raw_euronext(raw_euronext)
        logger.info("Extracting unique companies from Euronext data")
        companies_euronext = make_subset_of_companies_euronext(raw_euronext)
        logger.info(f"Adding market information to Euronext companies ({len(companies_euronext)} rows)")
        companies_euronext = add_market_column_euronext(companies_euronext)
        logger.info("Normalizing Euronext companies dataframe")
        companies_euronext = make_normalized_dataframe_euronext(companies_euronext)

    handle_companies(companies_euronext, companies_bousorama, raw_boursorama, raw_euronext, db)
    
    if not raw_euronext.empty:
        handle_daystocks(raw_euronext, raw_boursorama, db)
    else:
        logger.info("Skipping daystocks processing for Euronext data as no data is available")

    handle_stocks(raw_boursorama, db)


@timer_decorator
def store_files(years:list[str], db:TSDB):
    global full_companies_db_dataframe
    
    # Initialize the global DataFrame that will store all companies
    full_companies_db_dataframe = pd.DataFrame(
        columns=[
            "id", "name", "mid", "symbol", "isin", "boursorama", "euronext", 
        ],
    )

    logger.info(f"Starting ETL process for years {years}")

    if not is_data_present():
        logger.error("Data not present")
        raise ValueError("Data not present")

    verify_db_state(db)

    for year in years:
        load_year(year, db)

    logger.info("ETL process finished successfully.")
    
    
    

     




if __name__ == '__main__':
    print("Go Extract Transform and Load")
    pd.set_option('display.max_columns', None)  # usefull for dedugging
    db = tsdb.TimescaleStockMarketModel('bourse', 'ricou', 'db', 'monmdp')        # inside docker
    # db = tsdb.TimescaleStockMarketModel('bourse', 'ricou', 'localhost', 'monmdp') # outside docker
    years = ["2020"]
    print("Start Extract Transform and Load")
    store_files(years, db)
    print("Done Extract Transform and Load")
