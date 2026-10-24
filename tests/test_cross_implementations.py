"""The diffractometer geometry in other NaNIsaac repos, run on the same inputs as qorageom.

Each sibling implementation is taken from its source file by name (``ast``), not imported,
because those files are scripts that import GUI and data libraries at the top. The sibling
repos are looked up under ``$NANISAAC_ROOT``, or ``$DEV/NaNIsaac`` when that is unset, and a
test is skipped when its source is not there, so CI without the sibling repos skips them.

Agreement is asserted. Disagreement is recorded with ``xfail(strict=True)`` and a reason, so it
stays visible and the test fails once the other implementation is fixed.

Compared quantities, all for the central detector pixel:

- 2θ from detector (gamma, delta): CXD ``gam_del.py`` and the ``xrd_tools`` scripts against the
  34idc detector rotation in :class:`qorageom.Diffractometer`.
- |q| from (gamma, delta): the ``xrd_tools`` scripts.
- d from 2θ and energy: ``xrd_tools/src/physics.py``.
- Sample rotation matrices: ``9c-sim/sample.py``.
- q in the sample frame for a two-circle (omega, 2θ) setup: ``rocking_xrd_sim/sim.py``
  (xrayutilities).
"""

import ast
import os
from pathlib import Path

import numpy as np
import pytest

from qorageom import (
    Diffractometer,
    detector_pose,
    get_diffractometer,
    pixel_to_q_lab,
    q_to_tth,
    rotation_matrix,
    tth_to_d,
    tth_to_q,
    wavelength,
)

ROOT = Path(os.environ.get("NANISAAC_ROOT") or Path(os.environ.get("DEV", ".")) / "NaNIsaac")
BEAM = np.array([0.0, 0.0, 1.0])


def load(relative, *names, **namespace):
    """Execute the named top-level functions, classes and assignments of a sibling source file."""
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} not found under {ROOT}")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = []
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef | ast.ClassDef)
            and node.name in names
            or isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
        ):
            wanted.append(node)
    found = {n.name if hasattr(n, "name") else n.targets[0].id for n in wanted}
    assert found == set(names), f"{relative}: missing {set(names) - found}"
    scope = {"np": np, **namespace}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 - runs trusted sibling-repo source
    return scope


def two_theta_34idc(gamma, delta):
    """2θ [deg] of the direct beam after the 34idc detector arm (delta about y, gamma about -x)."""
    arm = get_diffractometer("34idc").detector_rotation
    return np.array(
        [
            np.rad2deg(np.arccos(np.clip((arm([d, g]) @ BEAM) @ BEAM, -1, 1)))
            for g, d in zip(np.ravel(gamma), np.ravel(delta), strict=True)
        ]
    ).reshape(np.shape(gamma))


# gamma and delta in degrees, away from the clip at cos(2θ) = ±1
GAMMA, DELTA = np.meshgrid(np.linspace(-20, 20, 9), np.linspace(2, 70, 12))
TTH = two_theta_34idc(GAMMA, DELTA)


def test_34idc_arm_obeys_cos_2theta_product():
    expected = np.rad2deg(np.arccos(np.cos(np.deg2rad(GAMMA)) * np.cos(np.deg2rad(DELTA))))
    np.testing.assert_allclose(TTH, expected, atol=1e-9)


def test_cxd_theory_delta_inverts_two_theta():
    # CXD moved the tool from the package to scripts/ (CXD#20); older checkouts have the old path
    path = next(
        (
            p
            for p in ("CXD/scripts/gam_del.py", "CXD/src/cxd/gam_del.py")
            if (ROOT / p).is_file()
            and "calculate_theory_delta" in (ROOT / p).read_text(encoding="utf-8")
        ),
        "CXD/scripts/gam_del.py",
    )
    f = load(path, "calculate_theory_delta")["calculate_theory_delta"]
    for tth, gamma, delta in zip(TTH.ravel(), GAMMA.ravel(), DELTA.ravel(), strict=True):
        assert f(np.array([gamma]), tth)[0] == pytest.approx(delta, abs=1e-6)


def test_xrd_tools_corrected_two_theta():
    f = load("xrd_tools/two_theta_y.py", "calculate_corrected_2theta")
    np.testing.assert_allclose(f["calculate_corrected_2theta"](GAMMA, DELTA, 0.0), TTH, atol=1e-9)


def test_xrd_tools_q_magnitude_matches_tth_to_q():
    f = load("xrd_tools/q_gam.py", "hc", "energy", "WAVELENGTH", "calculate_q_magnitude")
    lam = f["WAVELENGTH"]
    # hc is 12.39841662513396 here, CODATA 2018 12.39841984 in qorageom: 2.6e-7 apart
    assert lam == pytest.approx(wavelength(5.55), rel=1e-6)
    np.testing.assert_allclose(
        f["calculate_q_magnitude"](GAMMA, DELTA, 0.0), tth_to_q(TTH, lam), rtol=1e-9
    )


def test_xrd_tools_physical_model_delta_inverts_two_theta():
    f = load("xrd_tools/q_gam.py", "hc", "energy", "WAVELENGTH", "physical_model_delta")
    lam = f["WAVELENGTH"]
    for tth, gamma, delta in zip(TTH.ravel(), GAMMA.ravel(), DELTA.ravel(), strict=True):
        d = tth_to_d(tth, lam)
        assert f["physical_model_delta"](gamma, d, 0.0) == pytest.approx(delta, abs=1e-6)


@pytest.mark.filterwarnings("ignore:invalid value encountered in arccos")
def test_xrd_tools_trajectory_stays_on_one_two_theta():
    f = load("xrd_tools/delgam_traj.py", "WAVELENGTH", "calculate_trajectory")
    d = 1.2
    delta, gamma, tth = f["calculate_trajectory"](d, 0.0, 0.0)
    assert tth == pytest.approx(q_to_tth(2 * np.pi / d, f["WAVELENGTH"]))
    ok = np.isfinite(gamma)
    np.testing.assert_allclose(two_theta_34idc(gamma[ok], delta[ok]), tth, atol=1e-6)


def test_xrd_tools_specular_q_agrees_in_the_gamma_zero_plane():
    f = load("xrd_tools/delta_gamma_mapping.py", "WAVELENGTH", "calculate_q_specular")
    delta = np.linspace(5, 70, 8)
    _, _, q_total = f["calculate_q_specular"](delta, np.zeros_like(delta))
    np.testing.assert_allclose(q_total, tth_to_q(delta, f["WAVELENGTH"]), rtol=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="delta_gamma_mapping.calculate_q_specular takes |q| from delta alone, "
    "4π sin(δ/2)/λ, but the scattering angle is cos 2θ = cos δ cos γ; the two differ for gamma != 0",
)
def test_xrd_tools_specular_q_off_the_gamma_zero_plane():
    f = load("xrd_tools/delta_gamma_mapping.py", "WAVELENGTH", "calculate_q_specular")
    _, _, q_total = f["calculate_q_specular"](DELTA, GAMMA)
    np.testing.assert_allclose(q_total, tth_to_q(TTH, f["WAVELENGTH"]), rtol=1e-6)


def test_xrd_tools_tth2d_matches_tth_to_d():
    f = load("xrd_tools/src/physics.py", "hc_eV", "tth2d")
    tth = np.array([10.0, 35.0, 70.0, 120.0])
    energy_ev = 8000.0
    # tth2d returns nm (hc in eV nm), qorageom Å; hc differs in the 7th digit (1239.84173
    # against CODATA 2018 1239.84198)
    np.testing.assert_allclose(
        10 * f["tth2d"](tth, energy_ev), tth_to_d(tth, wavelength(energy_ev / 1000)), rtol=1e-6
    )


@pytest.mark.parametrize("angle", [-37.0, 0.0, 12.5, 90.0, 211.0])
def test_9c_sim_rotation_matrices_match_rotation_matrix(angle):
    sample = load("9c-sim/sample.py", "Sample", BeamlineState=object)["Sample"]()
    for name, axis in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1])):
        m = getattr(sample, f"_matrix_rotate_{name}")(angle)
        np.testing.assert_allclose(m[:3, :3], rotation_matrix(axis, angle), atol=1e-12)
        assert m[3, 3] == 1 and not m[:3, 3].any() and not m[3, :3].any()


@pytest.mark.parametrize("up_deg, side_deg", [(0.0, 0.0), (20.0, 0.0), (0.0, 35.0), (25.0, -40.0)])
def test_detector_pose_centre_matches_34idc_arm(up_deg, side_deg):
    """Two qorageom code paths for the same detector arm (gamma = up, delta = side)."""
    energy = 8.0
    center, _, _ = detector_pose(0.5, (0, 0, 1), (0, 1, 0), up_deg, side_deg)
    q_pose = pixel_to_q_lab(energy, (0, 0, 1), center, (1, 0, 0), (0, 1, 0), 55e-6, 0, 0)
    arm = get_diffractometer("34idc").detector_rotation([side_deg, up_deg])
    q_arm = 2 * np.pi * (arm @ BEAM - BEAM) / wavelength(energy)
    np.testing.assert_allclose(q_pose, q_arm, atol=1e-12)


@pytest.mark.parametrize("omega, tth", [(34.5, 69.0), (20.0, 69.0), (10.0, 40.0), (-5.0, 25.0)])
def test_rocking_sim_central_pixel_q_matches_two_circle_diffractometer(omega, tth):
    pytest.importorskip("xrayutilities")
    import xrayutilities as xu

    energy_ev = 5550.0
    sim = load(
        "rocking_xrd_sim/sim.py", "DET_9C", "geometry", xu=xu
    )  # geometry() builds the QConversion
    qconv, _, _, _, _, lam = sim["geometry"](energy_ev, {"npix": 255}, two_theta_deg=tth)
    q_xu = np.array(qconv.area(omega, tth, deg=True))[:, 127, 127]

    assert lam == pytest.approx(wavelength(energy_ev / 1000), rel=1e-12)
    d = Diffractometer("two-circle", ("x+",), ("om",), ("x+",), ("tth",), incident_axis=(0, 1, 0))
    q_lab = 2 * np.pi * (d.detector_rotation([tth]) @ np.array([0, 1.0, 0]) - [0, 1.0, 0]) / lam
    np.testing.assert_allclose(d.sample_rotation([omega]).T @ q_lab, q_xu, atol=1e-9)
