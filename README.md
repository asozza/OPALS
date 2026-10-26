# OPALS

Optimal Partitioning Along-with Land Suppression

Python port of the NEMO best domain decomposition (`ln_listonly = .true.` in `&nammpp`,
see `bestpartition` / `mpp_is_ocean` / `mpp_basesplit` in NEMO `mppini.F90`):
it lists the MPI decompositions that minimise the subdomain size with land-subdomain
suppression, reading only `domain_cfg.nc` (no need to run `nemo.exe`).

Requirements: `numpy`, `xarray` (with a netCDF backend).

```
python opals.py domain_cfg.nc -n 256          # list of best partitions, as in ocean.output
python opals.py domain_cfg.nc -n 256 -v       # also print jpni x jpnj for each entry
python opals.py domain_cfg.nc -n 256 --best   # only the best decomposition for 256 cores
```

`-n` plays the role of `max(mppsize, jpni*jpnj)`: the list explores up to about
`2 * n / (1 - land fraction)` subdomains.
