# QoraGeom

Detector geometry for X-ray scattering: computing the lab-frame q vector of an area-detector pixel from beam energy, sample-detector distance, detector angles and pixel size.

## Status

Early draft. The geometry code currently lives in `main.py`; the `qorageom` package under `src/` is still an empty stub. `main.py` defines:

- `detector_pose(distance_m, incident_dir, scatter_up, up_angle_deg, side_angle_deg)`: detector center position and the pixel x/y directions, from the sample-detector distance and two rotations of the detector arm (up, then sideways) relative to the incident beam,
- `pixel_to_q_lab(energy_kev, incident_dir, center_pos, pixel_x_dir, pixel_y_dir, pixel_size_m, px, py)`: q = k_f - k_i in 1/Å (lab frame) for pixel offset (px, py) from the detector center,
- `rotate(vector, axis, angle_deg)`: Rodrigues rotation.

## Usage

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NaNIsaac/QoraGeom.git
cd QoraGeom
uv run python main.py
```

This prints the detector pose and the q vector of pixel (1, 1) for an 8.04 keV beam, a 1 m detector distance and a 30° up angle.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
