# From diffraction pattern to strain


`examples/bcdi_pipeline.py` simulates a 16-voxel crystal (5 nm voxels, Au 111) with a localized Gaussian lattice displacement, computes its diffraction intensity, reconstructs it with [QoraPhase](https://github.com/NaNIsaac/QoraPhase) (4 random starts), aligns it to the amplitude of the true object, removes the phase ramp with `qoraphase.remove_ramp` and recovers the strain with `qorageom.strain` on the support eroded by one voxel. The crystal is simulated directly in lab axes, so no resampling is needed; the QoraPhase dependency lives in the `examples` group.

<!-- test -->
```bash
uv run --group examples python examples/bcdi_pipeline.py
```

```
reconstruction: 2 s, Fourier error 1.8e-07
true strain:  rms 1.77e-03, range [-4.14e-03, 4.76e-03]
estimator floor on the true phase: rms 6.75e-05
strain error: rms 4.29e-04, max 6.41e-04
```

This output was recorded in a local CPU run; reconstruction time varies by machine.

The "floor" line is the same estimator applied to the true phase, so it is the part of the error that comes from the 5 nm voxels and finite differences. The data are noiseless, and the global phase, translation and twin branch are fixed from the true amplitude.

