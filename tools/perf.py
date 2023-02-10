"""Perf
Collect the performance data under a folder and write to a csv database file.

Usage:
  perf.py CSV_FILE RESULT [--meta=meta_info_file]

CSV_FILE is the database file to store the performance data.
RESULT can be a folder or a file. If it's a folder, the script will find all
performance result and attach them to the database file.

Options:
  -h --help                     Show this screen.
  --meta=meta_info_file         Additional meta info that want to bring into database.
Examples:
    perf.py results.csv /tmp/result_output
    perf.py results.csv /tmp/result_output/test_detail.json
"""
import csv
import glob
import json
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Tuple, Union, Set

from docopt import docopt


class PerfIndicator:
    def __init__(self, indicator: str, fields: str):
        self._indicator = indicator[1:] if indicator.startswith(":") else indicator
        self.name, self.unit = self._indicator.split(":")
        self.fields = tuple(fields.split(":"))

    @property
    def field_names(self):
        return [f"{self.name}.{field}({self.unit})" for field in self.fields]

    def __repr__(self):
        return f"PerfIndicator({self.name},{self.unit})"


class PerfData:
    def __init__(self, indicator: PerfIndicator, name: str, data: str):
        self.indicator = indicator
        self._values = tuple(float(val) for val in data.strip().split(" "))
        self.name = name
        fields_cnt = len(self.indicator.fields)
        if fields_cnt != len(self._values):
            raise ValueError(
                f"Performance indicator has {fields_cnt} fields "
                f"but performance data {name} only has {len(self._values)} fields"
            )
        self._data = {
            filed: val for filed, val in zip(self.indicator.field_names, self._values)
        }

    @property
    def data(self) -> Dict[str, float]:
        return self._data

    def __repr__(self):
        indicator = f"<{self.indicator.name},{self.indicator.unit}>"
        return f"PerfData({self.name}{indicator} {self._values[0]},...)"


class PerfRecord:
    def __init__(self, test_suites: List[str], record: Dict):
        self._test_suites = test_suites
        self._record = record

    @cached_property
    def suite_name(self):
        return "/".join(self._test_suites)

    @cached_property
    def name(self):
        return self._record["name"].split("/")[0]

    @cached_property
    def sub_idx(self):
        try:
            idx = self._record["name"].split("/")[1]
            return int(idx)
        except (IndexError, TypeError):
            return 0

    @cached_property
    def full_name(self):
        return "/".join([self.suite_name, self.name])

    @property
    def scenario(self):
        type_param = self._record.get("type_param", "")
        value_param = self._record.get("value_param", "")
        return value_param or type_param

    @property
    def is_success(self):
        if "failures" in self._record:
            return False
        return True

    @cached_property
    def indicator_heads(self) -> Tuple[PerfIndicator]:
        items = self._record.items()
        return tuple(PerfIndicator(k, v) for k, v in items if k.startswith(":"))

    @cached_property
    def perf_data(self) -> Tuple[PerfData]:
        if not self.is_success:
            msg = f"unsuccessful test record {self.full_name} has no perf data"
            raise ValueError(msg)

        data_list = []
        for key, value in self._record.items():
            try:
                indicator_name, indicator_unit, data_name = key.split(":")
                if indicator := self._find_indicator(indicator_name, indicator_unit):
                    data_list.append(PerfData(indicator, data_name, value))
            except ValueError as e:
                if "not enough values to unpack" not in str(e):
                    raise e

        return tuple(data_list)

    def _find_indicator(self, name, unit) -> Union[PerfIndicator, None]:
        for indicator in self.indicator_heads:
            if name == indicator.name and unit == indicator.unit:
                return indicator
        return None

    def __repr__(self):
        return f"{self.full_name}::{self.sub_idx}::{self.scenario}"


class PerfResult:
    def __init__(self, perf_result_file):
        self._perf_result_file = Path(perf_result_file)
        with open(perf_result_file) as f:
            self._perf_data = json.load(f)

    @property
    def records(self):
        suites = self._perf_data["testsuites"]
        all_recs = []
        for s in suites:
            recs = list(self._parse_suites(s, tuple()))
            all_recs.extend(recs)
        return all_recs

    def _parse_suites(self, current_suite: Dict, parent_suits_name: Tuple):
        if next_suite := current_suite.get("testsuite", None):
            parents = *parent_suits_name, current_suite["name"]
            for rec in next_suite:
                yield from self._parse_suites(rec, parents)
        else:
            rec = PerfRecord(list(parent_suits_name), current_suite)
            yield rec


class PerfDataBase:
    def __init__(
        self,
        database_file: str,
        result_files: List[str],
        meta_info: Dict[str, str] = None,
    ):
        self._database_file = Path(database_file).resolve()
        self._create_new_database = not self._database_file.is_file()
        self._result_files = [Path(file) for file in result_files]
        self._meta_info = meta_info or {}

    DB_KEY_FIELDS = "test_id", "scenario", "kernel"

    @property
    def meta_info_heads(self) -> Tuple:
        return tuple(self._meta_info.keys())

    def write_to_database(self):
        all_recs = self._read_all_perf_records()
        if self._create_new_database:
            print(f"Create new database file at {self._database_file}")
            field_names = self._get_heads_for_new_csv_file(all_recs)
        else:
            print(f"Find database file at {self._database_file}, attach the data to it")
            field_names = self._existing_db_file_heads

        mode = "w" if self._create_new_database else "a"
        with open(self._database_file, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            if self._create_new_database:
                writer.writeheader()
            for rec in all_recs:
                for kernel_name, row in self._record_perf_data_to_rows(rec).items():
                    row = {key: row.get(key, None) for key in field_names}
                    if self._create_new_database:
                        writer.writerow(row)
                    else:
                        key = self.get_record_key(row)
                        if key in self._existing_db_test_ids:
                            writer.writerow(row)
                        else:
                            print(f"Skip unwanted perf data: {'::'.join(key)}")

    def _read_all_perf_records(self):
        all_recs = []
        for file in self._result_files:
            print(f"Process result file: {file.resolve()}")
            perf_result = PerfResult(file)
            for rec in perf_result.records:
                if rec.is_success:
                    all_recs.append(rec)
                else:
                    print(f"Warning: skip unsuccessful test result: {rec}")
        return all_recs

    @staticmethod
    def get_record_key(rec: Dict[str, str]) -> Tuple[str, ...]:
        return tuple(rec[key] for key in PerfDataBase.DB_KEY_FIELDS)

    def _get_heads_for_new_csv_file(self, all_records) -> List[str]:
        # different test records may have different performance indicator
        # this function is to combine all performance indicator used for csv heads.
        # use dict to remove duplicated fields and keep its order
        fields = {}
        for rec in all_records:
            for head in rec.indicator_heads:
                for name in head.field_names:
                    fields[name] = None
        heads = list(self.DB_KEY_FIELDS)
        heads.extend(self.meta_info_heads)
        heads.extend(fields.keys())
        return heads

    @cached_property
    def _existing_db_file_heads(self) -> List[str]:
        assert not self._create_new_database
        with open(self._database_file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            fieldnames = list(reader.fieldnames)
        for key in self.DB_KEY_FIELDS:
            if key not in fieldnames:
                raise ValueError(f"Not find key {key} in db file {self._database_file}")
        return fieldnames

    @cached_property
    def _existing_db_test_ids(self) -> Set[str]:
        assert not self._create_new_database
        with open(self._database_file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            ids = {self.get_record_key(row) for row in reader}
            return ids

    def _record_perf_data_to_rows(self, rec):
        common = self._create_common_perf_data_dict(rec)
        rows = {}
        for data in rec.perf_data:
            kernel_name = data.name
            try:
                row = rows[kernel_name]
            except KeyError:
                row = common.copy()
                row["kernel"] = kernel_name
                rows[kernel_name] = row
            row.update(data.data)
        return rows

    def _create_common_perf_data_dict(self, rec) -> Dict:
        common = {
            "test_id": rec.full_name,
            "scenario": rec.scenario,
        }
        common.update(self._meta_info)
        return common


def main(args_):
    database_file = args_["CSV_FILE"]

    result_file = Path(args_["RESULT"])
    if result_file.is_file():
        all_result_files = [result_file]
    elif result_file.is_dir():
        result_file = result_file.resolve()
        print(f"Search all test_detail.json under path {result_file}")
        all_result_files = glob.glob(
            f"{result_file}/**/test_detail.json", recursive=True
        )
    else:
        raise FileNotFoundError(result_file)

    meta_info = None
    if meta_info_file := args_["--meta"]:
        with open(meta_info_file, "r") as f:
            meta_info = json.load(f)

    db = PerfDataBase(database_file, all_result_files, meta_info)
    db.write_to_database()


if __name__ == "__main__":
    args_ = docopt(__doc__, version="Perf Tool")
    main(args_)
