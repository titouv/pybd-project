# PyBD Project

This project involves 

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