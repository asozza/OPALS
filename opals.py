#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Opals module

Python port of the NEMO "best domain decomposition" (ln_listonly = .true. in &nammpp),
i.e. bestpartition / mpp_is_ocean / mpp_basesplit / mpp_init_landprop from mppini.F90.
It lists the MPI decompositions that minimise the subdomain size while suppressing
land-only subdomains, without having to run nemo.exe.

Usage:
    python opals.py domain_cfg.nc -n 256
    python opals.py domain_cfg.nc -n 256 --best     # only the best partition for 256 cores

Author: Alessandro Sozza (CNR-ISAC)
Date: March 2025
"""

import argparse
import numpy as np
import xarray as xr


def read_bathymetry(domcfg, bdyfile=None):
    """
    Read the land-sea mask from the domain configuration file (readbot_strip)

    Args:
    - domcfg: path to domain_cfg.nc
    - bdyfile: optional path to a file containing bdy_msk

    Returns:
    - lloce: boolean array (Nj0glo, Ni0glo), True on ocean points
    - jperio: lateral boundary condition type
    - cfg: configuration name (if available)
    """

    ds = xr.open_dataset(domcfg)
    zbot = ds['bottom_level'].squeeze().values.astype(float)
    if bdyfile is not None:
        with xr.open_dataset(bdyfile) as dsb:
            zbot = zbot * dsb['bdy_msk'].squeeze().values
    jperio = int(ds['jperio']) if 'jperio' in ds else 0
    cfg = ds.attrs.get('cn_cfg', None)
    ds.close()

    return zbot > 0., jperio, cfg


def nint(x):
    """ Fortran NINT: round half away from zero """
    return int(np.sign(x) * np.floor(abs(x) + 0.5))


def mpp_basesplit(kiglo, kjglo, khls, knbi, knbj, jperio=0):
    """
    Lay out the global domain over processors (mpp_basesplit)

    Only the first row/column are returned, since all the subdomains
    of a given column (row) have the same i (j) size and position.

    Args:
    - kiglo, kjglo: global domain size
    - khls: halo width
    - knbi, knbj: number of subdomains along i and j
    - jperio: lateral boundary condition type

    Returns:
    - kimax, kjmax: largest subdomain size
    - kimppt, kjmppt: 1-based starting index of each column/row of subdomains
    - klci, klcj: size of each column/row of subdomains
    """

    i2hls = 2*khls
    kimax = ( kiglo - i2hls + (knbi-1) ) // knbi + i2hls
    kjmax = ( kjglo - i2hls + (knbj-1) ) // knbj + i2hls

    iresti = 1 + ( kiglo - i2hls - 1 ) % knbi
    irestj = 1 + ( kjglo - i2hls - 1 ) % knbj

    klci = np.full(knbi, kimax - 1, dtype=int)
    klci[:iresti] = kimax

    klcj = np.full(knbj, kjmax - 1, dtype=int)
    if jperio in (3, 4, 5, 6):
        # minimize the size of the last row to compensate for the north pole folding coast
        ijpjmin = 2 + 3*khls if jperio in (3, 4) else 1 + 3*khls
        irm = knbj - irestj                                  # total number of lines to be removed
        klcj[knbj-1] = max(ijpjmin, kjmax - irm)             # we must have jpj >= ijpjmin in the last row
        irm = irm - ( kjmax - klcj[knbj-1] )                 # remaining number of lines to remove
        irestj = knbj - 1 - irm
        klcj[max(irestj, 0):knbj-1] = kjmax - 1
    klcj[:max(irestj, 0)] = kjmax

    kimppt = np.ones(knbi, dtype=int)
    kimppt[1:] = 1 + np.cumsum(klci[:-1] - i2hls)
    kjmppt = np.ones(knbj, dtype=int)
    kjmppt[1:] = 1 + np.cumsum(klcj[:-1] - i2hls)

    return kimax, kjmax, kimppt, kjmppt, klci, klcj


def mpp_init_landprop(lloce):
    """ Proportion of land points in the surface land-sea mask (mpp_init_landprop) """

    return 1.0 - np.count_nonzero(lloce) / lloce.size


def mpp_is_ocean(lloce, jperio, inbi, inbj):
    """
    Check for a mpi domain decomposition inbi x inbj which subdomains,
    including 1 halo, contain at least 1 ocean point (mpp_is_ocean)

    Args:
    - lloce: boolean land-sea mask (Nj0glo, Ni0glo)
    - jperio: lateral boundary condition type
    - inbi, inbj: number of subdomains along i and j

    Returns:
    - ldisoce: boolean array (inbj, inbi), True if the subdomain contains ocean
    """

    Nj0glo, Ni0glo = lloce.shape
    _, _, iimppt, ijmppt, ijpi, ijpj = mpp_basesplit(Ni0glo, Nj0glo, 0, inbi, inbj, jperio)
    inboce = np.zeros((inbj, inbi), dtype=int)

    # work with Fortran-like 1-based indices on a (iny+1, inx+1) array, index 0 unused
    inx = Ni0glo + 2
    for iarea in range(1, inbj + 1):
        iny = ijpj[iarea-1] + 2                              # strip size + 1 halo on each direction
        strip = np.zeros((iny + 1, inx + 1), dtype=bool)     # strip[j, i]
        inry = iny - (iarea == 1) - (iarea == inbj)          # number of points to read in y-direction
        isty = 1 + (iarea == 1)                              # read from the first or the second line?
        kjstr = ijmppt[iarea-1] - 2 + isty
        strip[isty:inry+isty, 2:inx] = lloce[kjstr-1:kjstr-1+inry, :]

        if iarea == 1:                                       # the first line was not read
            if jperio in (2, 7):                             #   north-south periodicity
                strip[1, 2:inx] = lloce[Nj0glo-1, :]
            else:                                            #   closed boundary
                strip[1, 2:inx] = False
        if iarea == inbj:                                    # the last line was not read
            if jperio in (2, 7):                             #   north-south periodicity
                strip[iny, 2:inx] = lloce[0, :]
            elif jperio in (3, 4):                           #   north-pole folding T-pivot
                strip[iny, 2] = strip[iny-2, 2]
                for ji in range(3, inx):
                    strip[iny, ji] = strip[iny-2, inx-ji+2]
                for ji in range(inx//2 + 2, inx):
                    strip[iny-1, ji] = strip[iny-1, inx-ji+2]
            elif jperio in (5, 6):                           #   north-pole folding F-pivot
                strip[iny-1, inx//2+1] = strip[iny-1, inx//2]
                strip[iny-1, inx-1] = strip[iny-1, 2]
                for ji in range(2, inx):
                    strip[iny, ji] = strip[iny-1, inx-ji+1]
            else:                                            #   closed boundary
                strip[iny, 2:inx] = False

        # first and last column were not read
        if jperio in (1, 4, 6, 7):                           # east-west periodicity
            strip[:, 1] = strip[:, inx-1]
            strip[:, inx] = strip[:, 2]
        else:                                                # closed boundary
            strip[:, 1] = False
            strip[:, inx] = False

        # number of ocean points per column of the strip, then per subdomain
        icol = np.concatenate(([0], np.cumsum(strip[1:, 1:].sum(axis=0))))
        i0 = iimppt                                          # columns iimppt .. iimppt+ijpi+1
        i1 = iimppt + ijpi + 1
        inboce[iarea-1, :] = icol[i1] - icol[i0-1]

    return inboce != 0


def bestpartition(lloce, jperio, knbij, nn_hls=1, ldlist=False):
    """
    Find the best domain decomposition for NEMO (bestpartition)

    Args:
    - lloce: boolean land-sea mask (Nj0glo, Ni0glo)
    - jperio: lateral boundary condition type
    - knbij: number of MPI processes
    - nn_hls: halo width
    - ldlist: if True, return the list of the best partitions (ln_listonly)

    Returns:
    - if ldlist: list of dicts (nb_cores, land, percentage, size, iszi, iszj, jpni, jpnj)
                 sorted by decreasing number of ocean cores
    - otherwise: (knbi, knbj, knbcnt) with knbcnt the number of land subdomains
    """

    Nj0glo, Ni0glo = lloce.shape
    # global domain including halos: used as "larger than any subdomain" default
    # (nemo.exe uses jpiglo*jpjglo+1, otherwise the 1x1 partition would be discarded)
    jpiglo, jpjglo = Ni0glo + 2*nn_hls, Nj0glo + 2*nn_hls

    # get the proportion of land point over the global domain
    zpropland = mpp_init_landprop(lloce)
    inbij = nint( knbij / ( 1.0 - zpropland ) )

    # define the largest possible value for jpni*jpnj
    inbijmax = inbij*2 if ldlist else inbij

    # get the list of knbi that gives a smaller jpimax than knbi-1
    # get the list of knbj that gives a smaller jpjmax than knbj-1
    inbi0, iszi0, inbj0, iszj0 = [], [], [], []
    isziref = iszjref = jpiglo*jpjglo + 1
    for ji in range(1, inbijmax + 1):
        iszitst = ( Ni0glo + (ji-1) ) // ji + 2*nn_hls
        if iszitst < isziref:
            isziref = iszitst
            inbi0.append(ji)
            iszi0.append(isziref)
        iszjtst = ( Nj0glo + (ji-1) ) // ji + 2*nn_hls
        if iszjtst < iszjref:
            iszjref = iszjtst
            inbj0.append(ji)
            iszj0.append(iszjref)

    # combine these 2 lists to get all possible knbi*knbj <= inbijmax (j outer loop, as in Fortran)
    cand = [(ni, nj, si, sj)
            for nj, sj in zip(inbj0, iszj0)
            for ni, si in zip(inbi0, iszi0)
            if ni*nj <= inbijmax]

    # extract only the partitions which reduce the subdomain size in comparison with smaller partitions
    best = {}
    for c in cand:                                           # first occurrence of the min size for each ni*nj
        nij, szij = c[0]*c[1], c[2]*c[3]
        if nij not in best or szij < best[nij][2]*best[nij][3]:
            best[nij] = c
    parts = []
    iszij = jpiglo*jpjglo + 1
    for nij in sorted(best):
        c = best[nij]
        if c[2]*c[3] < iszij:
            parts.append(c)
            iszij = c[2]*c[3]

    if ldlist:
        res = []
        ni, nj = parts[-1][:2]
        inbijold = np.count_nonzero(mpp_is_ocean(lloce, jperio, ni, nj))
        for ni, nj, si, sj in parts[-2::-1]:
            inbij = np.count_nonzero(mpp_is_ocean(lloce, jperio, ni, nj))
            if inbij < inbijold:
                res.append(dict(nb_cores=inbij, land=ni*nj - inbij,
                                percentage=(ni*nj - inbij) / (ni*nj) * 100.,
                                size=si*sj, iszi=si, iszj=sj, jpni=ni, jpnj=nj))
                inbijold = inbij
        return res

    # start from the end of the list (smaller subdomains) while the number of ocean subdomains exceeds the number of procs
    for ni, nj, _, _ in parts[::-1]:
        inbij = np.count_nonzero(mpp_is_ocean(lloce, jperio, ni, nj))
        if inbij <= knbij:
            return ni, nj, ni*nj - inbij

    return None


def print_list(res):
    """ Print the list of best partitions with the same format as NEMO ocean.output """

    print('   list of the best partitions including land supression')
    print('   -----------------------------------------------------')
    print(' ')
    for r in res:
        print('nb_cores oce: {:6d}, land domains excluded: {:6d} ({:4.1f}%), '
              'largest oce domain: {:9d} ( {:6d} x {:6d} )'.format(
                  r['nb_cores'], r['land'], r['percentage'], r['size'], r['iszi'], r['iszj']))
    print(' ')
    print('   -----------------------------------------------------------')


def main():

    parser = argparse.ArgumentParser(description='NEMO best domain decomposition (ln_listonly) in python')
    parser.add_argument('domcfg', help='domain configuration file (domain_cfg.nc)')
    parser.add_argument('-n', '--ncores', type=int, required=True,
                        help='number of MPI processes, i.e. max(mppsize, jpni*jpnj)')
    parser.add_argument('--hls', type=int, default=1, help='halo width nn_hls (default: 1)')
    parser.add_argument('--bdy', default=None, help='optional file containing bdy_msk')
    parser.add_argument('--best', action='store_true',
                        help='print only the best decomposition for ncores instead of the list')
    parser.add_argument('-v', '--verbose', action='store_true', help='also print jpni x jpnj in the list')
    args = parser.parse_args()

    lloce, jperio, cfg = read_bathymetry(args.domcfg, args.bdy)
    nn_hls = max(1, args.hls)
    Nj0glo, Ni0glo = lloce.shape
    print(f'       Configuration = {cfg}' if cfg else '')
    print(f'       Ni0glo = {Ni0glo:12d}')
    print(f'       Nj0glo = {Nj0glo:12d}')
    print(f'       jperio = {jperio:12d}')
    print(' ')

    if args.best:
        ni, nj, icnt = bestpartition(lloce, jperio, args.ncores, nn_hls)
        print(f'   The best domain decomposition {ni} x {nj} with {icnt} land subdomains')
        print(f'      - uses a total of {ni*nj - icnt} mpi process')
        return

    res = bestpartition(lloce, jperio, args.ncores, nn_hls, ldlist=True)
    print_list(res)
    if args.verbose:
        print(' ')
        for r in res:
            print(f"nb_cores oce: {r['nb_cores']:6d}  ->  jpni = {r['jpni']:4d}, jpnj = {r['jpnj']:4d}")


if __name__ == '__main__':
    main()
