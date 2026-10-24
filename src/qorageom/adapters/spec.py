"""SPEC (Certified Scientific Software) scan files: the parser and the scan geometry they give.

An adapter: it reads a file format and calls the geometry core, which never imports it.
"""

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from qorageom.detector import get_detector
from qorageom.diffractometer import _normalize_motor, get_diffractometer
from qorageom.geometry import energy_to_kev
from qorageom.scan_geometry import ScanGeometry, rocking_scan_geometry

__all__ = ["ScanAxis", "Spec", "SpecScan", "geometry_from_spec"]

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
                current_scan = SpecScan(
                    scan_no=scan_no, command=command, motor_names=list(self.motor_names)
                )
                self.scans[scan_no] = current_scan
                continue

            if current_scan is None:
                if line.startswith("#O") and line[2:3].isdigit():
                    parts = line.split(None, 1)
                    if len(parts) > 1:
                        self.motor_names.extend(parts[1].split())
                else:
                    tag = line.split(" ", 1)[0]
                    self.file_metadata[tag] = line[len(tag) :].strip()
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
                current_scan.metadata[tag] = line[len(tag) :].strip()

    def get_scan(self, scan_no: int) -> SpecScan:
        if not self.scans:
            self.read()
        try:
            return self.scans[scan_no]
        except KeyError:
            raise KeyError(f"Scan {scan_no} not found in {self.filepath}") from None


_COMMON_OVERRIDES = {"energy", "distance_m", "scanmot", "scanmot_del"}


def _missing(scan: SpecScan, what: str, override: str):
    return ValueError(f"{what} not found in scan {scan.scan_no}; pass {override}=... to set it")


def geometry_from_spec(
    scan: SpecScan,
    diffractometer="34idc",
    detector=None,
    shape=None,
    binning=(1, 1, 1),
    **overrides,
) -> ScanGeometry:
    """Geometry of a rocking scan from the values in its SPEC header.

    Keyword overrides win over the file, as in cohere: the circle readings in degrees by the
    motor mnemonics of the diffractometer (``th``, ``chi``, ``phi``, ``delta``, ``gamma`` for
    34-ID-C), their offsets as ``<mnemonic>_offset`` (``chi_offset`` defaults to 90 there),
    ``energy`` (keV or eV), ``distance_m``, and ``scanmot`` with ``scanmot_del`` (step per frame
    in the motor's unit) when the scan command is not an ``ascan``/``dscan``. ``detector``
    defaults to ``#UIMDET``. ``shape`` is the (N1, N2, N3) of the volume, ``binning`` the
    detector binning of the data and the number of frames averaged per rocking step.

    The scan rocks one sample circle of the diffractometer, or the energy; the basis is
    ``Diffractometer.rocking_basis``, with the detector orientation of the registry.
    """
    diff = get_diffractometer(diffractometer)
    motors = diff.sample_motors + diff.detector_motors
    known = _COMMON_OVERRIDES | set(motors) | {f"{m}_offset" for m in diff.sample_motors}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"unknown override(s) {sorted(unknown)}, known: {sorted(known)}")

    det_name = detector or scan.detector
    if det_name is None:
        raise _missing(scan, "detector (#UIMDET)", "detector")
    det = get_detector(det_name)
    if shape is None:
        raise ValueError("pass shape=(N1, N2, N3), the shape of the volume to describe")

    readings = {}
    for mne in motors:
        if mne in overrides:
            readings[mne] = overrides[mne]
            continue
        try:
            readings[mne] = scan.motor(mne)
        except KeyError:
            raise _missing(scan, f"motor {mne!r}", mne) from None
    offsets = tuple(
        overrides.get(f"{mne}_offset", default)
        for mne, default in zip(
            diff.sample_motors,
            diff.sample_offsets_deg or (0.0,) * len(diff.sample_motors),
            strict=True,
        )
    )

    energy_raw = overrides.get("energy", scan.motor("energy") if _has(scan, "energy") else None)
    if energy_raw is None:
        raise _missing(scan, "energy", "energy")
    distance = overrides.get("distance_m", scan.detector_distance_m)
    if distance is None:
        raise _missing(scan, "detector distance (camdist or sdd)", "distance_m")

    axis = scan.scan_axis
    motor = overrides.get("scanmot", axis.motor if axis else None)
    step = overrides.get("scanmot_del", axis.step if axis else None)
    if motor is None or step is None:
        raise _missing(
            scan, "scan axis (the command is not an ascan or dscan)", "scanmot and scanmot_del"
        )
    wanted = _normalize_motor(motor)
    rock = next((m for m in diff.sample_motors if _normalize_motor(m) == wanted), None)
    if rock is None and wanted != "energy":
        raise ValueError(
            f"cannot rock about {motor!r}; supported: {list(diff.sample_motors)} and energy"
        )

    energy_kev, step_kev = energy_to_kev(energy_raw, step)
    return rocking_scan_geometry(
        shape,
        energy_kev,
        distance,
        [readings[m] for m in diff.sample_motors],
        [readings[m] for m in diff.detector_motors],
        rock or "energy",
        step if rock else step_kev,
        diffractometer=diff,
        detector=det,
        binning=binning,
        sample_offsets_deg=offsets,
    )


def _has(scan: SpecScan, name: str) -> bool:
    try:
        scan.motor(name)
    except KeyError:
        return False
    return True
