from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from qorageom.spec import Spec, SpecScan, _xml_to_dict
import xml.etree.ElementTree as ET


def test_parses_scan_metadata(spec_file: Path):
    spec = Spec(spec_file)

    assert set(spec.scans.keys()) == {1, 2}
    assert spec.motor_names == ["mot_a", "mot_b", "mot_c", "mot_d", "mot_e"]

    scan1 = spec.get_scan(1)
    assert scan1.scan_no == 1
    assert scan1.command == "ascan mot_a 0 1 10 1"
    assert scan1.columns == ["col_a", "col_b", "col_c"]
    assert scan1.time == datetime(2024, 1, 1, 12, 34, 56)


def test_data_lines_parsed_as_floats(spec_file: Path):
    scan = Spec(spec_file).get_scan(1)
    assert scan.data == [
        [0.0, 1.0, 2.0],
        [0.5, 1.5, 2.5],
        [1.0, 2.0, 3.0],
    ]


def test_to_dataframe_returns_columns(spec_file: Path):
    df = Spec(spec_file).get_scan(1).to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["col_a", "col_b", "col_c"]
    assert len(df) == 3
    assert df.iloc[1]["col_b"] == 1.5


def test_to_dataframe_empty_when_no_data():
    scan = SpecScan(scan_no=99, command="noop", motor_names=[])
    assert scan.to_dataframe().empty


def test_motors_zips_strict(spec_file: Path):
    scan = Spec(spec_file).get_scan(1)
    assert scan.motors == {
        "mot_a": 0.1, "mot_b": 0.2, "mot_c": 0.3,
        "mot_d": 0.4, "mot_e": 0.5,
    }


def test_motor_names_are_per_scan_copies(spec_file: Path):
    spec = Spec(spec_file)
    scan = spec.get_scan(1)
    spec.motor_names.append("mutated")
    assert "mutated" not in scan.motor_names


def test_uxml_dict_decoded(spec_file: Path):
    scan = Spec(spec_file).get_scan(1)
    assert scan.sdd == 123.45
    assert scan.beam_center == {"x": 512, "y": 256}
    rois = scan.rois
    assert isinstance(rois, list) and len(rois) == 2
    assert rois[0]["name"] == "roi1"
    assert rois[0]["x"] == 10


def test_uxml_caches_after_first_access(spec_file: Path):
    scan = Spec(spec_file).get_scan(1)
    first = scan.uxml
    second = scan.uxml
    assert first is second


def test_uxml_empty_when_no_fragments():
    scan = SpecScan(scan_no=1, command="x", motor_names=[])
    assert scan.uxml == {}
    assert scan.sdd is None
    assert scan.beam_center == {"x": 0, "y": 0}
    assert scan.rois == []


def test_uxml_parse_error_falls_back_to_raw(caplog):
    scan = SpecScan(scan_no=7, command="x", motor_names=[])
    scan._raw_uxml.append("<unclosed>")
    with caplog.at_level("WARNING"):
        result = scan.uxml
    assert result == {"raw": ["<unclosed>"]}
    assert any("scan 7" in rec.message for rec in caplog.records)


def test_get_scan_unknown_raises_with_path(spec_file: Path):
    spec = Spec(spec_file)
    with pytest.raises(KeyError, match=str(spec_file.name)):
        spec.get_scan(999)


def test_lazy_load_with_method_chaining(spec_file: Path):
    spec = Spec(spec_file, lazy=True)
    assert spec.scans == {}
    spec.read()
    assert 1 in spec.scans


def test_get_scan_triggers_lazy_read(spec_file: Path):
    spec = Spec(spec_file, lazy=True)
    scan = spec.get_scan(2)
    assert scan.command.startswith("ascan mot_b")


def test_with_context_loads_when_empty(spec_file: Path):
    with Spec(spec_file, lazy=True) as spec:
        assert 1 in spec.scans



def test_xml_to_dict_handles_negative_int():
    elem = ET.fromstring("<v>-5</v>")
    assert _xml_to_dict(elem) == -5
    assert isinstance(_xml_to_dict(elem), int)


def test_xml_to_dict_handles_float_and_string():
    assert _xml_to_dict(ET.fromstring("<v>1.5</v>")) == 1.5
    assert _xml_to_dict(ET.fromstring("<v>hello</v>")) == "hello"
    assert _xml_to_dict(ET.fromstring("<v></v>")) == ""


def test_xml_to_dict_repeated_tags_become_list():
    elem = ET.fromstring("<r><a>1</a><a>2</a><a>3</a></r>")
    assert _xml_to_dict(elem) == {"a": [1, 2, 3]}


def _scan(command: str, **kwargs) -> SpecScan:
    return SpecScan(scan_no=1, command=command, motor_names=[], **kwargs)


def test_scan_axis_from_fixture(spec_file: Path):
    axis = Spec(spec_file).get_scan(2).scan_axis
    assert axis is not None
    assert (axis.motor, axis.start, axis.stop, axis.npts) == ("mot_b", -1.0, 1.0, 5)
    assert axis.step == pytest.approx(0.4)


@pytest.mark.parametrize("command", [
    "mesh a 0 1 2 b 0 1 2 1",
    "ascan Phi 1 2",
    "ascan Phi 1 two 10 1",
    "ascan Phi 1 2 0 1",
    "",
])
def test_scan_axis_none_for_unsupported_commands(command: str):
    assert _scan(command).scan_axis is None


def test_scan_axis_dscan_negative_step():
    axis = _scan("dscan Phi 0.5 -0.5 20 2").scan_axis
    assert axis.step == pytest.approx(-0.05)


def test_motor_lookup_ignores_case_and_aliases():
    scan = _scan("", _raw_p_values=[37.0, -1.0, 5.0])
    scan.motor_names = ["Delta", "Gamma", "Th_eta"]
    assert scan.motor("del") == 37.0
    assert scan.motor("GAMMA") == -1.0
    assert scan.motor("theta") == 5.0
    with pytest.raises(KeyError):
        scan.motor("chi")


def test_detector_and_roi_from_header_lines(tmp_path: Path):
    path = tmp_path / "det.spec"
    path.write_text(
        """#F det.spec
#O0 Delta

#S 1 ascan Phi 0 1 10 1
#P0 3.0
#UIMDET Timepix:
#UIMR5 100 120 64 64
#L a
1
""",
        encoding="utf-8",
    )
    scan = Spec(path).get_scan(1)
    assert scan.detector == "Timepix"
    assert scan.roi == [100, 120, 64, 64]


def test_detector_and_roi_none_when_absent(spec_file: Path):
    scan = Spec(spec_file).get_scan(1)
    assert scan.detector is None
    assert scan.roi is None
