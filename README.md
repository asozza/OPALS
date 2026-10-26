# OPALS

**Optimal Partitioning Along-with Land Suppression**

Python port of the NEMO best domain decomposition (`ln_listonly = .true.` in `&nammpp`,
see `bestpartition` / `mpp_is_ocean` / `mpp_basesplit` in NEMO `mppini.F90`):
it lists the MPI decompositions that minimise the subdomain size with land-subdomain
suppression, reading only `domain_cfg.nc` (no need to run `nemo.exe`).

Requirements: `numpy`, `xarray` (with a netCDF backend), `matplotlib` (only for `--plot`).

## Usage

```
python opals.py domain_cfg.nc -n 256                          # list of best partitions, as in ocean.output
python opals.py domain_cfg.nc -n 256 -v                       # also print jpni x jpnj for each entry
python opals.py domain_cfg.nc -n 1024 --cores-per-node 128    # also the best partition for each number of nodes
python opals.py domain_cfg.nc -n 256 --best                   # best decomposition for 256 cores + &nammpp block
python opals.py domain_cfg.nc --jpni 16 --jpnj 16 -n 256      # check a given decomposition (wasting CPU?)
python opals.py domain_cfg.nc -n 256 --best --plot map.png    # map of the subdomains, land ones shaded
python opals.py domain_cfg.nc --save-mask orca2_mask.npz      # save the land-sea mask in a few kB
```

`-n` plays the role of `max(mppsize, jpni*jpnj)`: the list explores up to about
`2 * n / (1 - land fraction)` subdomains.

`--cores-per-node` picks, for each number of nodes, the partition with the smallest subdomains
that fits in it, and reports the idle cores left on the nodes (e.g. for XIOS or other components).

A `.npz` mask written by `--save-mask` can be used everywhere in place of `domain_cfg.nc`.
`domain_cfg.nc` files are too large for the repository (see `.gitignore`).

## Tests

```
python -m unittest -v test_opals
```

`orca2_mask.npz` and `eorca1_mask.npz` are the land-sea masks of ORCA2 and eORCA1,
`orca2_ref.txt` is the list printed by `nemo.exe` for ORCA2 with 256 cores, reproduced exactly.
