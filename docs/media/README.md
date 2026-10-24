# README media

`tth_map.png` is drawn by `make_figure.py` from this commit's code:

```sh
uv run --with matplotlib docs/media/make_figure.py
```

It calls `tth_map` for a 1000 x 1000 detector of 75 µm pixels, 150 mm from the sample, with the
PONI at the centre and the arm at 0° and at 25°, converts 2θ to |q| at 8.04 keV with `tth_to_q`,
and draws |q| as colour and 2θ as 5° contours. No random numbers are involved. Running it again
gives the same file byte for byte (checked on Windows, matplotlib 3.11.2).
