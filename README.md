# MLflow migration data
Migration of  mlflow experiments and runs metadata from the file system to the database.

Use utility **DataMigrator.migrate_data** for migrate experiments, runs, metrics, params, tags (of runs).

Parameters:
* config_path="conf/db.json"

  Database configuration file.
  

* root_log_dir="./mlruns"

  The root directory of metadata in the file system.


* queries_file=None

  The name of the file to generate insert queries.


* init_tables=True, 

    Create mlflow tables (the full list of tables is located in the DataMigrator._TABLES property).


* clean_all_tables=False

    If true, will delete all records from all mlflow tables before writing.
