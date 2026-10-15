# QoraGeom

Detector geometry for X-ray scattering: compute the lab-frame q vector of a detector pixel from beam energy, detector distance and detector angles. Early stage; the code currently lives in `main.py`.

## Usage

```
uv run main.py
```

`main.py` provides:

- `rotate(vector, axis, angle_deg)` - Rodrigues rotation
- `detector_pose(distance_m, incident_dir, scatter_up, up_angle_deg, side_angle_deg)` - detector center position and pixel x/y directions
- `pixel_to_q_lab(energy_kev, incident_dir, center_pos, pixel_x_dir, pixel_y_dir, pixel_size_m, px, py)` - q (1/Angstrom) of pixel (px, py) relative to the detector center

Example from `main()`:

```python
center_pos, px_dir, py_dir = detector_pose(
    distance_m=1.0, incident_dir=[1, 0, 0], scatter_up=[0, 1, 0],
    up_angle_deg=30, side_angle_deg=0,
)
q = pixel_to_q_lab(8.04, [1, 0, 0], center_pos, px_dir, py_dir, 55e-6, px=1, py=1)
```

## Requirements

Python >= 3.14, numpy.
