import mlflow
import psycopg2
import json
import yaml
import os
import re
from tqdm import tqdm

from mlflow.entities import RunStatus
from mlflow.entities import SourceType


def migrate_data(config_path="conf/db.json", root_log_dir="./mlruns", queries_file=None,
                 init_tables=True, clean_all_tables=False):
    migrator = DataMigrator(config_path)
    migrator.write_insert_queries(root_log_dir, queries_file)
    migrator.send_queries(init_tables=init_tables, clean_all_tables=clean_all_tables)


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

        self._clear_file(self._queries_file)

        print("The follow 'experiment/run' were processed:")
        with open(self._queries_file, "a") as sink:
            for experiment_id in os.listdir(root_log_dir):
                if experiment_id.startswith("."):
                    continue

                with open(f"{root_log_dir}/{experiment_id}/meta.yaml") as ef:
                    experiment = yaml.load(ef, Loader=yaml.FullLoader)
                experiment_insert = self._get_experiment_insert(**experiment)
                sink.write(experiment_insert)

                for run_uuid in os.listdir(f"{root_log_dir}/{experiment_id}"):
                    if run_uuid == 'meta.yaml' or run_uuid.startswith("."):
                        continue
                    with open(f"{root_log_dir}/{experiment_id}/{run_uuid}/meta.yaml") as rf:
                        run = yaml.load(rf, Loader=yaml.FullLoader)
                    if not run:
                        continue
                    run_insert = self._get_run_insert(**run)
                    sink.write(run_insert)
                    print(f"{experiment_id}/{run_uuid}")

                    path_to_run_folder = f"{root_log_dir}/{experiment_id}/{run_uuid}"

                    self._write_run_metrics_inserts(path_to_run_folder, run_uuid, sink)
                    self._write_run_params_inserts(path_to_run_folder, run_uuid, sink)
                    self._write_run_tags_inserts(path_to_run_folder, run_uuid, sink)

    def init_tables(self):
        dummy_experiment_name = "__DataMigrator_dummy_experiment"
        mlflow.set_tracking_uri(self._get_uri())
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

    def _get_uri(self):
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

    @staticmethod
    def _clear_file(path):
        with open(path, 'w'):
            pass

    @staticmethod
    def _get_experiment_insert(**experiment):
        query = f"""
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
        return re.sub(r'\s{2,}', "", query) + "\n"

    @staticmethod
    def _get_run_insert(**run):
        if run['end_time'] is None:
            run['end_time'] = "NULL"

        query = f"""
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
                {run['experiment_id']}
            );
        """
        return re.sub(r'\s{2,}', "", query) + "\n"

    @staticmethod
    def _get_metric_insert(is_nan=False, **metric):
        if metric['val'] == "nan":
            metric['val'] = -1.
        elif metric['val'] == "inf":
            metric['val'] = -9.

        query = f"""
            INSERT INTO metrics (
                key, 
                value, 
                timestamp, 
                run_uuid,
                step,
                is_nan
            ) VALUES ( 
                '{metric['metric']}', 
                {metric['val']}, 
                {metric['timestamp']}, 
                '{metric['run_uuid']}',
                {metric['step']},
                {is_nan}
            );
        """
        return re.sub(r'\s{2,}', "", query) + "\n"

    @staticmethod
    def _get_param_insert(**param):
        query = f"""
            INSERT INTO params (
                key, 
                value, 
                run_uuid
            ) VALUES ( 
                '{param['param']}', 
                '{param['val'].strip()}', 
                '{param['run_uuid']}' 
            );
        """
        return re.sub(r'\s{2,}', "", query) + "\n"

    @staticmethod
    def _get_tag_insert(**tag):
        query = f"""
                INSERT INTO tags (
                    key, 
                    value, 
                    run_uuid
                ) VALUES ( 
                    '{tag['tag']}', 
                    '{tag['val'].strip()}', 
                    '{tag['run_uuid']}' 
                );
            """
        return re.sub(r'\s{2,}', "", query) + "\n"

    def _write_run_tags_inserts(self, path_to_run_folder, run_uuid, sink_file):
        for tag in os.listdir(f"{path_to_run_folder}/tags"):
            with open(f"{path_to_run_folder}/tags/{tag}") as tagf:
                val = tagf.read().strip()
                if not val:
                    continue
                tag_insert = self._get_tag_insert(
                    tag=tag,
                    val=val,
                    run_uuid=run_uuid,
                )
                sink_file.write(tag_insert)

    def _write_run_params_inserts(self, path_to_run_folder, run_uuid, sink_file):
        for param in os.listdir(f"{path_to_run_folder}/params"):
            with open(f"{path_to_run_folder}/params/{param}") as pf:
                val = pf.read().strip()
                if not val:
                    continue
                param_insert = self._get_param_insert(
                    param=param,
                    val=pf.read(),
                    run_uuid=run_uuid,
                )
                sink_file.write(param_insert)

    def _write_run_metrics_inserts(self, path_to_run_folder, run_uuid, sink_file):
        for path, _, metrics in os.walk(f"{path_to_run_folder}/metrics"):
            for metric in metrics:
                with open(os.path.join(path, metric)) as mf:
                    for line in mf.readlines():
                        timestamp, val, step = line.split()
                        metric_insert = self._get_metric_insert(
                            metric=metric,
                            val=val,
                            timestamp=timestamp,
                            run_uuid=run_uuid,
                            step=step,
                        )
                        sink_file.write(metric_insert)
            # if metric_insert:
            #     lastest_metruc_insert = metric_insert.replace(" metrics ", " latest_metrics ", 1)
            #     sink_file.write(lastest_metruc_insert)
