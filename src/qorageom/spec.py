"""SPEC (Certified Scientific Software) scan file parser: #O/#P motors, #D time, #UXML."""
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from qorageom.geometry import energy_to_kev

logger = logging.getLogger(__name__)


def _xml_to_dict(element: ET.Element) -> Any:
    if len(element) == 0:
        val = element.text.strip() if element.text else ""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            return val

    result = {}
    if element.attrib:
        result.update(element.attrib)

    for child in element:
        child_data = _xml_to_dict(child)
        tag = child.tag

        if tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(child_data)
        else:
            result[tag] = child_data
    return result


_MOTOR_ALIASES = {"del": "delta", "gam": "gamma", "th": "theta", "en": "energy"}


def _normalize_motor(name: str) -> str:
    name = re.sub(r"[^a-z0-9]", "", name.lower())
    return _MOTOR_ALIASES.get(name, name)


@dataclass(frozen=True)
class ScanAxis:
    motor: str
    start: float
    stop: float
    intervals: int  # SPEC's 4th argument; the scan has intervals + 1 points

    @property
    def step(self) -> float:
        return (self.stop - self.start) / self.intervals


@dataclass
class SpecScan:
    scan_no: int
    command: str
    motor_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    columns: list[str] = field(default_factory=list)
    data: list[list[float]] = field(default_factory=list)
    time: datetime | None = None
    _raw_p_values: list[float] = field(default_factory=list, repr=False)
    _raw_uxml: list[str] = field(default_factory=list, repr=False)
    _uxml_dict: dict[str, Any] | None = field(default=None, repr=False, init=False)

    @property
    def motors(self) -> dict[str, float]:
        return dict(zip(self.motor_names, self._raw_p_values, strict=True))

    @property
    def uxml(self) -> dict[str, Any]:
        if self._uxml_dict is not None:
            return self._uxml_dict
        if not self._raw_uxml:
            self._uxml_dict = {}
            return {}

        xml_string = "<root>\n" + "\n".join(self._raw_uxml) + "\n</root>"
        try:
            self._uxml_dict = _xml_to_dict(ET.fromstring(xml_string))
        except ET.ParseError as e:
            logger.warning("XML parsing error on scan %s: %s", self.scan_no, e)
            self._uxml_dict = {"raw": self._raw_uxml}
        return self._uxml_dict

    def motor(self, name: str) -> float:
        """Start-of-scan position of a motor; case, punctuation and del/gam aliases are ignored."""
        wanted = _normalize_motor(name)
        for key, value in self.motors.items():
            if _normalize_motor(key) == wanted:
                return value
        raise KeyError(f"Motor {name} not found in scan {self.scan_no}")

    @property
    def scan_axis(self) -> ScanAxis | None:
        """Motor and range of an ``ascan``/``dscan``; None for other scan commands."""
        parts = self.command.split()
        if len(parts) < 5 or parts[0] not in ("ascan", "dscan"):
            return None
        try:
            axis = ScanAxis(parts[1], float(parts[2]), float(parts[3]), int(parts[4]))
        except ValueError:
            return None
        return axis if axis.intervals > 0 else None

    @property
    def energy_kev(self) -> float | None:
        """Photon energy in keV from the ``Energy`` motor (keV or eV in the file); None if absent."""
        try:
            return float(energy_to_kev(self.motor("energy"))[0])
        except KeyError:
            return None

    @property
    def detector_distance_m(self) -> float | None:
        """Sample-detector distance in metres from the ``camdist`` motor, else ``sdd``; both mm."""
        try:
            return self.motor("camdist") / 1000
        except KeyError:
            return self.sdd / 1000 if self.sdd is not None else None

    @property
    def detector(self) -> str | None:
        """Detector name from ``#UIMDET``, without the trailing colon."""
        name = self.metadata.get("#UIMDET")
        return name.removesuffix(":") if name else None

    @property
    def roi(self) -> list[int] | None:
        """Detector ROI numbers from ``#UIMR5``."""
        try:
            return [int(n) for n in self.metadata["#UIMR5"].split()]
        except (KeyError, ValueError):
            return None

    def to_dataframe(self):
        """Scan data as a pandas DataFrame with the #L column names (needs pandas)."""
        import pandas as pd  # optional dependency, only this method needs it

        if not self.columns or not self.data:
            return pd.DataFrame()
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def sdd(self) -> float | None:
        return self.uxml.get("ad_detector", {}).get("sample_detector_distance", None)

    @property
    def rois(self) -> list[dict[str, Any]]:
        roi_data = self.uxml.get("ad_regions_of_interest", {}).get("roi", [])
        if isinstance(roi_data, dict):
            return [roi_data]
        return roi_data

    @property
    def beam_center(self) -> dict[str, float]:
        """Center pixel of the first detector ROI; {0, 0} when the scan has no ROI."""
        if not self.rois:
            return {"x": 0, "y": 0}
        roi = self.rois[0]
        return {
            "x": roi["min_x"] + roi["size_x"] // 2,
            "y": roi["min_y"] + roi["size_y"] // 2,
        }


class Spec:
    def __init__(self, file: str | Path, lazy: bool = False):
        self.filepath = Path(file)
        self.file_metadata: dict[str, Any] = {}
        self.motor_names: list[str] = []
        self.scans: dict[int, SpecScan] = {}
        if not lazy:
            self.read()

    def read(self) -> "Spec":
        with self.filepath.open("r", encoding="utf-8") as f:
            self._parse(f)
        return self

    def __enter__(self) -> "Spec":
        if not self.scans:
            self.read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # read() already closed the file; the context manager only mirrors open().
        pass

    def _parse(self, file: TextIO):
        current_scan = None

        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#S "):
                parts = line.split(maxsplit=2)
                scan_no = int(parts[1])
                command = parts[2] if len(parts) > 2 else ""
                current_scan = SpecScan(scan_no=scan_no, command=command, motor_names=list(self.motor_names))
                self.scans[scan_no] = current_scan
                continue

            if current_scan is None:
                if line.startswith("#O") and line[2:3].isdigit():
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        self.motor_names.extend(parts[1].split())
                else:
                    tag = line.split(" ", 1)[0]
                    self.file_metadata[tag] = line[len(tag):].strip()
            elif not line.startswith("#"):
                try:
                    current_scan.data.append(list(map(float, line.split())))
                except ValueError:
                    pass
            elif line.startswith("#D "):
                current_scan.time = datetime.strptime(line[3:], "%a %b %d %H:%M:%S %Y")
            elif line.startswith("#L "):
                current_scan.columns = line[3:].split()
            elif line.startswith("#P") and line[2:3].isdigit():
                parts = line.split(None, 1)
                if len(parts) > 1:
                    current_scan._raw_p_values.extend(float(x) for x in parts[1].split())
            elif line.startswith("#UXML"):
                xml_content = re.sub(r"^#UXML\s*", "", line)
                if xml_content:
                    current_scan._raw_uxml.append(xml_content)
            else:
                tag = line.split(" ", 1)[0]
                current_scan.metadata[tag] = line[len(tag):].strip()

    def get_scan(self, scan_no: int) -> SpecScan:
        if not self.scans:
            self.read()
        try:
            return self.scans[scan_no]
        except KeyError:
            raise KeyError(f"Scan {scan_no} not found in {self.filepath}") from None
