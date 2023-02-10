import csv
import json
from pathlib import Path
from shutil import copy

import pytest
import shlex

from docopt import docopt

from perf import main
import perf


_ = pytest
fixture_folder = Path(__file__).parent / "fixtures"
test_detail_file = fixture_folder / "test_detail.json"
meta_info_file = fixture_folder / "meta_info.json"
db_ref_file = fixture_folder / "db_ref.csv"
existing_db_file = fixture_folder / "existing_db.csv"
attach_db_ref_file = fixture_folder / "attach_db_ref.csv"


@pytest.fixture()
def test_detail():
    file = Path(__file__).parent / "fixtures/test_detail.json"
    with open(file) as f:
        return json.load(f)


@pytest.fixture()
def last_rec():
    records = perf.PerfResult(test_detail_file).records
    last_rec = records[-1]
    assert last_rec.is_success
    return last_rec


@pytest.fixture()
def tmp_db_file(tmp_path):
    tmp_file = tmp_path / "tmp_db.csv"
    if tmp_file.is_file():
        tmp_file.unlink()
    print(f"tmp_db_file is {tmp_file}")
    return tmp_file


class TestAPP:
    def test_docopt_example(self):
        doc = perf.__doc__
        for line in reversed(doc.splitlines()):
            if "Examples" in line:
                return
            if example := line.strip():
                print(f"test example command line:")
                print(example)
                argv = shlex.split(example)
                arguments = docopt(doc, argv[1:])
                assert arguments

    def test_create_empty_database(self, tmp_db_file):
        argvs = [str(tmp_db_file), str(test_detail_file), f"--meta={meta_info_file}"]
        self._run_app(argvs)
        self._compare_to_ref(tmp_db_file, db_ref_file)

    def test_without_meta_info_file(self, tmp_db_file):
        argvs = [str(tmp_db_file), str(test_detail_file)]
        self._run_app(argvs)
        assert tmp_db_file.is_file()

    def test_search_test_detail_file_in_folder(self, tmp_db_file):
        folder = test_detail_file.parent.parent
        argvs = [str(tmp_db_file), str(folder), f"--meta={meta_info_file}"]
        self._run_app(argvs)
        assert tmp_db_file.is_file()
        self._compare_to_ref(tmp_db_file, db_ref_file)

    def test_headers_and_perf_data_entry_should_be_same_when_attaching_to_a_db_file(
        self, tmp_db_file
    ):
        file_to_attach = tmp_db_file
        copy(existing_db_file, file_to_attach)
        argvs = [
            str(file_to_attach),
            str(test_detail_file),
            f"--meta={meta_info_file}",
        ]
        self._run_app(argvs)
        with open(existing_db_file) as old, open(file_to_attach) as new:
            old_csv = csv.DictReader(old)
            new_csv = csv.DictReader(new)
            assert list(old_csv.fieldnames) == list(new_csv.fieldnames)

            def to_id(row):
                return row["test_id"], row["scenario"], row["kernel"]

            old_items = {to_id(row) for row in old_csv}
            new_items = {to_id(row) for row in new_csv}
            assert old_items == new_items
        self._compare_to_ref(file_to_attach, attach_db_ref_file)

    @staticmethod
    def _compare_to_ref(act_file, ref_file):
        with open(act_file) as act, open(ref_file) as ref:
            act_lines = act.readlines()
            ref_lines = ref.readlines()
            assert len(act_lines) == len(ref_lines)
            for l1, l2 in zip(act_lines, ref_lines):
                assert l1.strip() == l2.strip()

    @staticmethod
    def _run_app(argvs):
        doc = perf.__doc__
        arguments = docopt(doc, argvs)
        main(arguments)


class TestPerf:
    def test_parse_records(self):
        records = perf.PerfResult(test_detail_file).records
        assert len(records) > 10
        first, last = records[0], records[-1]

        assert first.full_name == "nerf/mlp_fusion_test/cm_interop_tf32"
        assert first.sub_idx == 0
        assert not first.is_success

        assert last.full_name == "nerf/mlp_fusion_test/esimd_fp16"
        assert last.sub_idx == 2
        assert last.is_success

    def test_perf_indicators(self, last_rec):
        first = last_rec.indicator_heads[0]
        assert first.name == "kernel_time"
        assert first.unit == "ms"
        assert len(first.fields) >= 4

    def test_perf_data(self, last_rec):
        first = last_rec.perf_data[0]
        assert first.name == "fused_gemm"
        assert first.indicator.name == "kernel_time"
        assert first.indicator.unit == "ms"
        assert first.data == {
            "kernel_time.minimum(ms)": 1.1,
            "kernel_time.maximum(ms)": 5.5,
            "kernel_time.median(ms)": 3.3,
            "kernel_time.mean(ms)": 4.4,
        }

    def test_failed_test_record(self):
        perf_ = perf.PerfResult(test_detail_file)
        rec = next(rec for rec in perf_.records if not rec.is_success)
        with pytest.raises(ValueError):
            _ = rec.perf_data


class TestPerfDataBase:
    def test_write_to_empty_database(self, tmp_path):
        tmp_file = tmp_path / "abc.csv"
        if tmp_file.is_file():
            tmp_file.unlink()
        meta_info = {
            "time": "2023.1.1",
            "driver ver": "v12.3",
            "compiler ver": "dpcpp.1234",
        }
        db = perf.PerfDataBase(str(tmp_file), [str(test_detail_file)], meta_info)
        db.write_to_database()
