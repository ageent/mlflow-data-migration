import mlflow
import psycopg2
import json
import yaml
import os
from tqdm import tqdm

from mlflow.entities import RunStatus
from mlflow.entities import SourceType


class DataMigrator:
    _TABLES = (
        "alembic_version",
        "experiment_tags",
        "tags",
        "latest_metrics",
        "metrics",
        "model_versions",
        "params",
        "registered_models",
        "runs",
        "experiments",
    )

    def __init__(self, config_path="conf/db.json"):
        with open(config_path) as f:
            self._conf = json.load(f)
        self._queries_file = "queries"
        self._counter_value_on_last_commit = 0

    def write_insert_queries(self, root_log_dir="./mlruns", queries_file=None):
        if queries_file is not None:
            self._queries_file = queries_file

        # clear file
        with open(self._queries_file, 'w'):
            pass

        with open(self._queries_file, "a") as sink:
            for experiment_id in os.listdir(root_log_dir):
                if experiment_id == '.trash':
                    continue

                with open(f"{root_log_dir}/{experiment_id}/meta.yaml", "r") as ef:
                    experiment = yaml.load(ef, Loader=yaml.FullLoader)
                experiment_insert = f"""
                    INSERT INTO experiments (
                        experiment_id, 
                        name, 
                        artifact_location, 
                        lifecycle_stage
                    ) VALUES (
                        {experiment['experiment_id']}, 
                        '{experiment['name']}', 
                        '{experiment['artifact_location']}', 
                        '{experiment['lifecycle_stage']}'
                    );
                """
                sink.write(" ".join(experiment_insert.split()) + "\n")

                for run_uuid in os.listdir(f"{root_log_dir}/{experiment_id}"):
                    if run_uuid in ['meta.yaml', ".ipynb_checkpoints"]:
                        continue
                    with open(f"{root_log_dir}/{experiment_id}/{run_uuid}/meta.yaml", "r") as rf:
                        run = yaml.load(rf, Loader=yaml.FullLoader)
                    if not run:
                        continue
                    if run['end_time'] is None:
                        run['end_time'] = "NULL"
                    run_insert = f"""
                        INSERT INTO runs (
                            run_uuid, 
                            name, 
                            source_type, 
                            source_name, 
                            entry_point_name, 
                            user_id, 
                            status, 
                            start_time, 
                            end_time, 
                            source_version, 
                            lifecycle_stage, 
                            artifact_uri, 
                            experiment_id
                        ) VALUES ( 
                            '{run['run_uuid']}', 
                            '{run['name']}', 
                            '{SourceType.to_string(run['source_type'])}', 
                            '{run['source_name']}', 
                            '{run['entry_point_name']}', 
                            '{run['user_id']}', 
                            '{RunStatus.to_string(run['status'])}', 
                            {run['start_time']}, 
                            {run['end_time']}, 
                            '{run['source_version']}', 
                            '{run['lifecycle_stage']}', 
                            '{run['artifact_uri']}', 
                            {experiment_id}
                        );
                    """
                    sink.write(" ".join(run_insert.split()) + "\n")
                    print(f"{experiment_id}/{run_uuid}")

                    path_to_run_folder = f"{root_log_dir}/{experiment_id}/{run_uuid}"

                    # Metrics
                    inner_folders_of_metric = os.listdir(f"{path_to_run_folder}/metrics")
                    if inner_folders_of_metric:
                        inner_folder_of_metric = inner_folders_of_metric[0]
                        path_to_metric = f"{path_to_run_folder}/metrics/{inner_folder_of_metric}"
                        for metric in os.listdir(path_to_metric):
                            with open(f"{path_to_metric}/{metric}", "r") as mf:
                                line = mf.readline()
                                while line:
                                    timestamp, val, step = line.split()
                                    if val == "nan":
                                        val = -1.
                                    elif val == "inf":
                                        val = -9.
                                    is_nan = False
                                    metric_insert = f"""
                                        INSERT INTO metrics (
                                            key, 
                                            value, 
                                            timestamp, 
                                            run_uuid,
                                            step,
                                            is_nan
                                        ) VALUES ( 
                                            '{metric}', 
                                            {val}, 
                                            {timestamp}, 
                                            '{run_uuid}',
                                            {step},
                                            {is_nan}
                                        );
                                    """
                                    sink.write(" ".join(metric_insert.split()) + "\n")
                                    line = mf.readline()
                        if metric_insert:
                            lastest_metruc_insert = metric_insert.replace(" metrics ", " latest_metrics ", 1)
                            sink.write(" ".join(lastest_metruc_insert.split()) + "\n")

                    # Params

                    for param in os.listdir(f"{path_to_run_folder}/params"):
                        with open(f"{path_to_run_folder}/params/{param}", "r") as pf:
                            line = pf.readline()
                            while line:
                                param_insert = f"""
                                    INSERT INTO params (
                                        key, 
                                        value, 
                                        run_uuid
                                    ) VALUES ( 
                                        '{param}', 
                                        '{line.strip()}', 
                                        '{run_uuid}' 
                                    );
                                """
                                sink.write(" ".join(param_insert.split()) + "\n")
                                line = pf.readline()

                    # Tags
                    for tag in os.listdir(f"{path_to_run_folder}/tags"):
                        with open(f"{path_to_run_folder}/tags/{tag}", "r") as tagf:
                            line = tagf.read()
                            tag_insert = f"""
                                INSERT INTO tags (
                                    key, 
                                    value, 
                                    run_uuid
                                ) VALUES ( 
                                    '{tag}', 
                                    '{line.strip()}', 
                                    '{run_uuid}' 
                                );
                            """
                            sink.write(" ".join(tag_insert.split()) + "\n")

    def init_tables(self):
        dummy_experiment_name = "__DataMigrator_dummy_experiment"
        mlflow.set_tracking_uri(self._gen_uri())
        mlflow.set_experiment(dummy_experiment_name)
        self._delete_record(table="experiments", field="name", field_value=f"'{dummy_experiment_name}'")

    def clean_all_tables(self):
        self._clean_tables(tables=self._TABLES)

    def get_db_cursor(cursor_foo):
        def wrapper(self, *args, **kwargs):
            with psycopg2.connect(
                    dbname=self._conf['database'],
                    user=self._conf['username'],
                    password=self._conf['username'],
                    host=self._conf['host'],
                    port=self._conf['port'],
            ) as conn:
                with conn.cursor() as cursor:
                    kwargs['conn'] = conn
                    kwargs['cursor'] = cursor
                    return cursor_foo(self, *args, **kwargs)

        return wrapper

    @get_db_cursor
    def send_queries(self, start_query_num=None, init_tables=True, clean_all_tables=False, **kwargs):
        """

        :param clean_all_tables:
        :param init_tables:
        :param start_query_num: from 0
        :return:
        """
        conn = kwargs['conn']
        cursor = kwargs['cursor']
        if init_tables:
            self.init_tables()
        if clean_all_tables:
            self.clean_all_tables()

        for _ in tqdm(self._queries_iterator(start_query_num, conn, cursor)):
            pass

    def _queries_iterator(self, start_query_num, conn, cursor):
        if start_query_num is None:
            start_query_num = self._counter_value_on_last_commit

        with open(self._queries_file, "r") as f:
            for i in range(start_query_num):
                query = f.readline()
                yield
            while query:
                try:
                    cursor.execute(query)
                except Exception as e:
                    print(f"The exception raised when querying: {query}")
                    raise e
                if i % 999 == 0:
                    conn.commit()
                    self._counter_value_on_last_commit = i
                i += 1
                query = f.readline()
                yield

    def _gen_uri(self):
        prefix = self._conf['dialect']
        if self._conf['driver']:
            prefix = f"{prefix}+{self._conf['driver']}"

        return f"{prefix}://{self._conf['username']}:{self._conf['password']}@{self._conf['host']}:{self._conf['port']}/{self._conf['database']}"

    @get_db_cursor
    def _delete_record(self, table, field, field_value, **kwargs):
        cursor = kwargs['cursor']
        cursor.execute(f"DELETE FROM {table} WHERE {field}={field_value};")

    @get_db_cursor
    def _clean_tables(self, tables, **kwargs):
        cursor = kwargs['cursor']
        for t in tables:
            cursor.execute(f"DELETE FROM {t};")
            mlflow.set_experiment()
