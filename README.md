# QoraGeom

Detector geometry for X-ray scattering: computing the lab-frame q vector of an area-detector pixel from beam energy, sample-detector distance, detector angles and pixel size.

## Status

Early stage. The repository holds only this README, the license and the citation file; no code has been published yet. The planned first version is a small NumPy module that:

- places the detector from its distance and two rotation angles relative to the incident beam,
- returns the pixel x/y directions on the detector plane,
- converts a pixel index to q (1/Å) in the lab frame for a given beam energy.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
