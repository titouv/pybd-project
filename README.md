# PyBD Project

This project involves analyzing 5 years of financial data from the Paris Stock Exchange. The data comes from two sources: Euronext (daily data) and Boursorama (10-minute intervals). The goal is to:

1. Read, clean and store the data efficiently in a TimescaleDB database
2. Create a dashboard using Dash to visualize and analyze the data, including:
   - Display stock prices on logarithmic scale (line or candlestick charts)
   - Compare multiple stocks with different colors
   - Show Bollinger Bands indicators
   - Display data tables with daily statistics (min, max, open, close, mean, std)
   - Additional custom analysis features

## Download dataset

You can download the dataset by running the following command:

```bash
python download.py
```

## Docker development

You need to have the dataset downloaded in the `data` folder.

You need for now to replace in the `docker/docker-compose.yml` file the path of the volumes
Replace `/home/ricou/bourse/data` and `/home/ricou/bourse/timescaledb` by the path of your local folder

```bash
cd docker/
make all
```
This command will build the docker images for the ETL and the dashboard, it will then start three containers:
- `db`: the database
- `etl`: the ETL process
- `dashboard`: the dashboard

You can then access the dashboard at `http://localhost:8050`

## Students

- Titouan Verhille
- Enguerrand Turcat
- Matthew Banawa
