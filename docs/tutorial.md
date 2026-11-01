# Convert a diffraction angle to |q|

This command runs in an isolated uv environment with the published package, without requiring
a project checkout. It uses photon energy 8.04 keV and 2θ = 30°.

<!-- test console -->
```console
$ uv run -q --no-project --with qorageom python -c "from qorageom import wavelength, tth_to_q, tth_to_d; lam = wavelength(8.04); print('wavelength: %.5f angstrom' % lam); print('q: %.5f inverse angstrom' % tth_to_q(30, lam)); print('d: %.5f angstrom' % tth_to_d(30, lam))"
wavelength: 1.54209 angstrom
q: 2.10909 inverse angstrom
d: 2.97909 angstrom
```

The q conversion uses |q| = 4π sin(θ)/λ; d is the Bragg plane spacing λ/(2 sin θ).
Angles passed to these conversion functions are in degrees. Other geometry functions have
different angle and length conventions: follow the [units guide](guides/geometry-and-io.md#units-and-conventions).
