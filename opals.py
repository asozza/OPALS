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
    python opals.py domain_cfg.nc -n 256 --cores-per-node 128   # add the best choice per number of nodes
    python opals.py domain_cfg.nc -n 256 --best                 # only the best partition for 256 cores
    python opals.py domain_cfg.nc --jpni 20 --jpnj 15           # check a given decomposition
    python opals.py domain_cfg.nc -n 256 --best --plot map.png  # map of the decomposition
    python opals.py domain_cfg.nc --save-mask orca2.npz         # small land-sea mask, usable instead of domain_cfg.nc

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
    - domcfg: path to domain_cfg.nc, or to a mask written by save_mask (.npz)
    - bdyfile: optional path to a file containing bdy_msk

    Returns:
    - lloce: boolean array (Nj0glo, Ni0glo), True on ocean points
    - jperio: lateral boundary condition type
    - cfg: configuration name (if available)
    """

    if str(domcfg).endswith('.npz'):
        with np.load(domcfg) as npz:
            shape = tuple(npz['shape'])
            lloce = np.unpackbits(npz['mask'])[:shape[0]*shape[1]].reshape(shape).astype(bool)
            cfg = str(npz['cfg']) or None
            return lloce, int(npz['jperio']), cfg

    ds = xr.open_dataset(domcfg)
    zbot = ds['bottom_level'].squeeze().values.astype(float)
    if bdyfile is not None:
        with xr.open_dataset(bdyfile) as dsb:
            zbot = zbot * dsb['bdy_msk'].squeeze().values
    jperio = int(ds['jperio']) if 'jperio' in ds else 0
    cfg = ds.attrs.get('cn_cfg', None)
    ds.close()

    return zbot > 0., jperio, cfg


def read_levels(domcfg):
    """
    Read the number of wet levels of each water column (bottom_level - top_level + 1)

    Args:
    - domcfg: path to domain_cfg.nc, or to a mask written by save_mask (.npz)

    Returns:
    - integer array (Nj0glo, Ni0glo), 0 on land, or None if not available
    """

    if str(domcfg).endswith('.npz'):
        with np.load(domcfg) as npz:
            return npz['levels'].astype(int) if 'levels' in npz else None

    with xr.open_dataset(domcfg) as ds:
        kbot = ds['bottom_level'].squeeze().values.astype(int)
        ktop = ds['top_level'].squeeze().values.astype(int) if 'top_level' in ds else np.ones_like(kbot)

    return np.where(kbot > 0, kbot - np.maximum(ktop, 1) + 1, 0)


def save_mask(filename, lloce, jperio, cfg=None, levels=None):
    """
    Save the land-sea mask as a small bit-packed .npz file (a few kB instead of the full domain_cfg.nc),
    optionally with the number of wet levels used by the load-balance diagnostics
    """

    extra = {} if levels is None else dict(levels=levels.astype(np.uint16))
    np.savez_compressed(filename, mask=np.packbits(lloce), shape=np.array(lloce.shape),
                        jperio=jperio, cfg=cfg or '', **extra)


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


def mpp_is_ocean(lloce, jperio, inbi, inbj, lxios=True):
    """
    Check for a mpi domain decomposition inbi x inbj which subdomains,
    including 1 halo, contain at least 1 ocean point (mpp_is_ocean)

    Args:
    - lloce: boolean land-sea mask (Nj0glo, Ni0glo)
    - jperio: lateral boundary condition type
    - inbi, inbj: number of subdomains along i and j
    - lxios: NEMO compiled with key_xios, which keeps the first subdomain of each row of land-only subdomains

    Returns:
    - ldisoce: boolean array (inbj, inbi), True if the subdomain contains ocean (or is kept for XIOS)
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

    ldisoce = inboce != 0
    if lxios:
        # XIOS decomposes the domain in bands and crashes if a band contains only suppressed
        # land subdomains: NEMO keeps the first subdomain of each row of land subdomains
        ldisoce[~ldisoce.any(axis=1), 0] = True

    return ldisoce


def bestpartition(lloce, jperio, knbij, nn_hls=1, ldlist=False, lxios=True):
    """
    Find the best domain decomposition for NEMO (bestpartition)

    Args:
    - lloce: boolean land-sea mask (Nj0glo, Ni0glo)
    - jperio: lateral boundary condition type
    - knbij: number of MPI processes
    - nn_hls: halo width
    - ldlist: if True, return the list of the best partitions (ln_listonly)
    - lxios: NEMO compiled with key_xios (see mpp_is_ocean)

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
    # (minimum subdomain size so that halos always address the neighbour inner domain,
    #  and V and F folding are outside of the southern halos)
    iszimin = iszjmin = 3*nn_hls
    if jperio in (3, 4):
        iszjmin = max(iszjmin, 2 + 3*nn_hls)
    elif jperio in (5, 6):
        iszjmin = max(iszjmin, 1 + 3*nn_hls)
    inbi0, iszi0, inbj0, iszj0 = [], [], [], []
    isziref = iszjref = jpiglo*jpjglo + 1
    for ji in range(1, inbijmax + 1):
        iszitst = ( Ni0glo + (ji-1) ) // ji + 2*nn_hls
        if iszitst < isziref and iszitst >= iszimin:
            isziref = iszitst
            inbi0.append(ji)
            iszi0.append(isziref)
        iszjtst = ( Nj0glo + (ji-1) ) // ji + 2*nn_hls
        if iszjtst < iszjref and iszjtst >= iszjmin:
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
    for c in cand:                                           # min size for each ni*nj, then the smaller
        nij, key = c[0]*c[1], (c[2]*c[3], c[2] + c[3])       # perimeter, then the first occurrence
        if nij not in best or key < (best[nij][2]*best[nij][3], best[nij][2] + best[nij][3]):
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
        inbijold = np.count_nonzero(mpp_is_ocean(lloce, jperio, ni, nj, lxios))
        for ni, nj, si, sj in parts[-2::-1]:
            inbij = np.count_nonzero(mpp_is_ocean(lloce, jperio, ni, nj, lxios))
            if inbij < inbijold:
                res.append(dict(nb_cores=inbij, land=ni*nj - inbij,
                                percentage=(ni*nj - inbij) / (ni*nj) * 100.,
                                size=si*sj, iszi=si, iszj=sj, jpni=ni, jpnj=nj))
                inbijold = inbij
        return res

    # start from the end of the list (smaller subdomains) while the number of ocean subdomains exceeds the number of procs
    for ni, nj, _, _ in parts[::-1]:
        inbij = np.count_nonzero(mpp_is_ocean(lloce, jperio, ni, nj, lxios))
        if inbij <= knbij:
            return ni, nj, ni*nj - inbij

    return None


def decomposition_info(lloce, jperio, jpni, jpnj, nn_hls=1, lxios=True):
    """
    Describe a given jpni x jpnj decomposition

    Returns a dict with the number of ocean/land subdomains, the largest subdomain
    size (jpimax, jpjmax, including halos) and the ocean mask of the subdomains (jpnj, jpni)
    """

    Nj0glo, Ni0glo = lloce.shape
    isoce = mpp_is_ocean(lloce, jperio, jpni, jpnj, lxios)
    jpimax, jpjmax, *_ = mpp_basesplit(Ni0glo + 2*nn_hls, Nj0glo + 2*nn_hls, nn_hls, jpni, jpnj, jperio)
    inbij = int(np.count_nonzero(isoce))

    return dict(jpni=jpni, jpnj=jpnj, nb_cores=inbij, land=jpni*jpnj - inbij,
                percentage=(jpni*jpnj - inbij) / (jpni*jpnj) * 100.,
                jpimax=jpimax, jpjmax=jpjmax, isoce=isoce)


def load_balance(lloce, jperio, jpni, jpnj, nn_hls=1, levels=None, lxios=True):
    """
    Load-balance diagnostics of a jpni x jpnj decomposition, based on the wet points
    in the inner domain (no halos) of each ocean subdomain

    NEMO computes on every point of a subdomain (land points are masked), so its cost
    scales with jpimax*jpjmax*jpk: the wet fraction measures how much of this work is
    done on ocean points. The imbalance (max / mean wet points per subdomain) matters for
    the parts whose cost scales with the wet points (e.g. sea ice, I/O, some diagnostics).

    Args:
    - lloce: boolean land-sea mask (Nj0glo, Ni0glo)
    - jperio: lateral boundary condition type
    - jpni, jpnj: number of subdomains along i and j
    - nn_hls: halo width
    - levels: optional number of wet levels per column (Nj0glo, Ni0glo): 3D wet points instead of 2D

    Returns:
    - dict with wet points per ocean subdomain (wmin, wmax, wmean), imbalance = wmax/wmean
      and wet_frac = wet points / computed points (inner domain of the ocean subdomains)
    """

    Nj0glo, Ni0glo = lloce.shape
    isoce = mpp_is_ocean(lloce, jperio, jpni, jpnj, lxios)
    _, _, iimppt, ijmppt, ijpi, ijpj = mpp_basesplit(Ni0glo + 2*nn_hls, Nj0glo + 2*nn_hls, nn_hls,
                                                     jpni, jpnj, jperio)
    # inner domain of each subdomain, 0-based in the (Nj0glo, Ni0glo) arrays
    i0, i1 = iimppt - 1, iimppt - 1 + ijpi - 2*nn_hls
    j0, j1 = ijmppt - 1, ijmppt - 1 + ijpj - 2*nn_hls

    zwet = lloce.astype(int) if levels is None else np.where(lloce, levels, 0)
    zcum = np.zeros((Nj0glo + 1, Ni0glo + 1), dtype=np.int64)       # integral image
    zcum[1:, 1:] = zwet.cumsum(axis=0).cumsum(axis=1)
    iwet = zcum[j1][:, i1] - zcum[j0][:, i1] - zcum[j1][:, i0] + zcum[j0][:, i0]   # (jpnj, jpni)
    iwet = iwet[isoce]

    nlev = 1 if levels is None else int(levels.max())
    icomp = isoce.sum() * (ijpi.max() - 2*nn_hls) * (ijpj.max() - 2*nn_hls) * nlev

    return dict(wmin=int(iwet.min()), wmax=int(iwet.max()), wmean=float(iwet.mean()),
                imbalance=float(iwet.max() / iwet.mean()), wet_frac=float(iwet.sum() / icomp) * 100.)


def best_per_node(res, ncpn, nextra=0):
    """
    For each number of nodes, pick the best partition of the list that fits in it

    The list is sorted by decreasing number of cores (i.e. increasing subdomain size),
    so the best partition on k nodes is the first one with nb_cores + nextra <= k*ncpn.
    Node counts that would not improve on k-1 nodes are skipped.

    Args:
    - res: list of best partitions from bestpartition(..., ldlist=True)
    - ncpn: number of cores per node
    - nextra: cores reserved on the same nodes for other components (XIOS, OIFS, runoff-mapper, ...)

    Returns:
    - list of dicts (nodes, idle, fill + the partition keys)
    """

    out = []
    knodes = -(-(res[0]['nb_cores'] + nextra) // ncpn)
    for k in range(1, knodes + 1):
        r = next((r for r in res if r['nb_cores'] + nextra <= k*ncpn), None)
        if r is not None and r['nb_cores'] + nextra > (k-1)*ncpn:
            nused = r['nb_cores'] + nextra
            out.append(dict(r, nodes=k, idle=k*ncpn - nused, fill=nused / (k*ncpn) * 100.))

    return out


def print_namelist(r):
    """ Print the &nammpp block and the number of MPI tasks for a given partition """

    print('!-----------------------------------------------------------------------')
    print('&nammpp        !   Massively Parallel Processing')
    print('!-----------------------------------------------------------------------')
    print(f"   jpni        = {r['jpni']:4d}   !  number of processors following i")
    print(f"   jpnj        = {r['jpnj']:4d}   !  number of processors following j")
    print('/')
    print(f"   ==> run nemo.exe on {r['nb_cores']} MPI tasks ({r['land']} land subdomains suppressed)")


def plot_decomposition(filename, lloce, jperio, jpni, jpnj, cfg=None, lxios=True):
    """
    Plot the land-sea mask with the jpni x jpnj subdomains on top:
    suppressed land subdomains are shaded, ocean subdomains are labelled with their MPI rank

    The subdomain boundaries are the ones used for land suppression (inner domain, no halos).
    """

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    Nj0glo, Ni0glo = lloce.shape
    _, _, iimppt, ijmppt, ijpi, ijpj = mpp_basesplit(Ni0glo, Nj0glo, 0, jpni, jpnj, jperio)
    isoce = mpp_is_ocean(lloce, jperio, jpni, jpnj, lxios)

    fig, ax = plt.subplots(figsize=(12, 12 * Nj0glo / Ni0glo + 0.8))
    ax.imshow(lloce, origin='lower', cmap=ListedColormap(['#e6dcc3', '#9ecae1']), vmin=0, vmax=1,
              extent=(0, Ni0glo, 0, Nj0glo), interpolation='nearest')

    labels = jpni*jpnj <= 600
    irank = 0
    for jj in range(jpnj):                                   # same numbering as mpp_getnum
        for ji in range(jpni):
            x0, y0 = iimppt[ji] - 1, ijmppt[jj] - 1
            if isoce[jj, ji]:
                ax.add_patch(Rectangle((x0, y0), ijpi[ji], ijpj[jj], fill=False, lw=0.6, ec='k'))
                if labels:
                    ax.text(x0 + ijpi[ji]/2, y0 + ijpj[jj]/2, str(irank), ha='center', va='center',
                            fontsize=max(4, min(9, 400 / max(jpni, jpnj) / 4)))
                irank += 1
            else:
                ax.add_patch(Rectangle((x0, y0), ijpi[ji], ijpj[jj], fc='0.45', ec='k', lw=0.6,
                                       alpha=0.75, hatch='//'))

    inbij = int(np.count_nonzero(isoce))
    title = f'{cfg}: ' if cfg else ''
    title += (f'{jpni} x {jpnj} = {jpni*jpnj} subdomains, {inbij} ocean MPI tasks, '
              f'{jpni*jpnj - inbij} land subdomains suppressed ({(jpni*jpnj - inbij) / (jpni*jpnj) * 100:.1f}%)')
    ax.set_title(title)
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    ax.set_xlim(0, Ni0glo)
    ax.set_ylim(0, Nj0glo)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def print_balance(res, lloce, jperio, nn_hls=1, levels=None, lxios=True):
    """ Print the load-balance diagnostics (see load_balance) of a list of partitions """

    cwet = '2D' if levels is None else '3D'
    print(f'   load balance ({cwet} wet points in the inner domain of the ocean subdomains)')
    print('   -----------------------------------------------------------')
    for r in res:
        b = load_balance(lloce, jperio, r['jpni'], r['jpnj'], nn_hls, levels, lxios)
        print(f"   jpni x jpnj = {r['jpni']:4d} x {r['jpnj']:4d}, nb_cores oce: {r['nb_cores']:6d}, "
              f"wet points min/mean/max: {b['wmin']:7d} / {b['wmean']:9.1f} / {b['wmax']:7d}, "
              f"imbalance: {b['imbalance']:5.2f}, wet fraction: {b['wet_frac']:5.1f}%")


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
    parser.add_argument('domcfg', help='domain configuration file (domain_cfg.nc) or mask saved with --save-mask')
    parser.add_argument('-n', '--ncores', type=int, default=None,
                        help='number of MPI processes, i.e. max(mppsize, jpni*jpnj)')
    parser.add_argument('--hls', type=int, default=1, help='halo width nn_hls (default: 1)')
    parser.add_argument('--bdy', default=None, help='optional file containing bdy_msk')
    parser.add_argument('--best', action='store_true',
                        help='print only the best decomposition for ncores instead of the list')
    parser.add_argument('--jpni', type=int, default=None, help='check this decomposition along i (with --jpnj)')
    parser.add_argument('--jpnj', type=int, default=None, help='check this decomposition along j (with --jpni)')
    parser.add_argument('--cores-per-node', type=int, default=None,
                        help='also print the best partition for each number of nodes')
    parser.add_argument('--extra-cores', type=int, default=0,
                        help='cores reserved on the same nodes for other components (with --cores-per-node)')
    parser.add_argument('--balance', action='store_true',
                        help='print load-balance diagnostics (3D wet points if the levels are available)')
    parser.add_argument('--plot', default=None, metavar='FILE',
                        help='plot the decomposition (--jpni/--jpnj, otherwise the best one for ncores)')
    parser.add_argument('--save-mask', default=None, metavar='FILE.npz',
                        help='save the land-sea mask in a small .npz file and exit')
    parser.add_argument('--no-xios', action='store_true',
                        help='NEMO compiled without key_xios: do not keep one subdomain in the rows of land subdomains')
    parser.add_argument('-v', '--verbose', action='store_true', help='also print jpni x jpnj in the list')
    args = parser.parse_args()

    lcheck = args.jpni is not None or args.jpnj is not None
    if lcheck and (args.jpni is None or args.jpnj is None):
        parser.error('--jpni and --jpnj must be given together')
    if args.ncores is None and not lcheck and not args.save_mask:
        parser.error('-n/--ncores is required (unless --jpni/--jpnj or --save-mask are given)')

    lloce, jperio, cfg = read_bathymetry(args.domcfg, args.bdy)
    nn_hls = max(1, args.hls)
    lxios = not args.no_xios
    levels = read_levels(args.domcfg) if args.balance or args.save_mask else None

    if args.save_mask:
        save_mask(args.save_mask, lloce, jperio, cfg, levels)
        print(f'   land-sea mask saved in {args.save_mask}')
        return

    Nj0glo, Ni0glo = lloce.shape
    if cfg:
        print(f'       Configuration = {cfg}')
    print(f'       Ni0glo = {Ni0glo:12d}')
    print(f'       Nj0glo = {Nj0glo:12d}')
    print(f'       jperio = {jperio:12d}')
    print(' ')

    if lcheck:
        # same diagnostics as mpp_init when jpni and jpnj are given in the namelist
        r = decomposition_info(lloce, jperio, args.jpni, args.jpnj, nn_hls, lxios)
        print(f"   The chosen domain decomposition {r['jpni']} x {r['jpnj']} with {r['land']} land subdomains")
        print(f"      - uses a total of {r['nb_cores']} mpi process")
        print(f"      - has mpi subdomains with a maximum size of (jpi = {r['jpimax']}, jpj = {r['jpjmax']}, "
              f"jpi*jpj = {r['jpimax']*r['jpjmax']})")
        ni, nj, _ = bestpartition(lloce, jperio, args.ncores or r['nb_cores'], nn_hls, lxios=lxios)
        b = decomposition_info(lloce, jperio, ni, nj, nn_hls, lxios)
        print(f"   The best domain decomposition {b['jpni']} x {b['jpnj']} with {b['land']} land subdomains")
        print(f"      - uses a total of {b['nb_cores']} mpi process")
        print(f"      - has mpi subdomains with a maximum size of (jpi = {b['jpimax']}, jpj = {b['jpjmax']}, "
              f"jpi*jpj = {b['jpimax']*b['jpjmax']})")
        if b['jpimax']*b['jpjmax'] < r['jpimax']*r['jpjmax'] or \
           (b['jpimax']*b['jpjmax'] == r['jpimax']*r['jpjmax'] and b['nb_cores'] < r['nb_cores']):
            print('    ---   YOU ARE WASTING CPU...   ---')
        else:
            print('   ==> you use the best mpi decomposition')
        if args.balance:
            print(' ')
            print_balance([r, b], lloce, jperio, nn_hls, levels, lxios)
        print(' ')
        print_namelist(r)
    elif args.best:
        ni, nj, _ = bestpartition(lloce, jperio, args.ncores, nn_hls, lxios=lxios)
        r = decomposition_info(lloce, jperio, ni, nj, nn_hls, lxios)
        print(f"   The best domain decomposition {ni} x {nj} with {r['land']} land subdomains")
        print(f"      - uses a total of {r['nb_cores']} mpi process")
        if args.balance:
            print(' ')
            print_balance([r], lloce, jperio, nn_hls, levels, lxios)
        print(' ')
        print_namelist(r)
    else:
        res = bestpartition(lloce, jperio, args.ncores, nn_hls, ldlist=True, lxios=lxios)
        print_list(res)
        if args.verbose:
            print(' ')
            for r in res:
                print(f"nb_cores oce: {r['nb_cores']:6d}  ->  jpni = {r['jpni']:4d}, jpnj = {r['jpnj']:4d}")
        if args.balance:
            print(' ')
            print_balance(res, lloce, jperio, nn_hls, levels, lxios)
        if args.cores_per_node:
            print(' ')
            print(f"   best partition for each number of nodes ({args.cores_per_node} cores per node"
                  + (f", {args.extra_cores} reserved for other components)" if args.extra_cores else ")"))
            print('   -----------------------------------------------------------')
            for r in best_per_node(res, args.cores_per_node, args.extra_cores):
                print(f"   nodes: {r['nodes']:4d}, nb_cores oce: {r['nb_cores']:6d}, idle cores: {r['idle']:4d} "
                      f"(fill {r['fill']:5.1f}%), jpni x jpnj = {r['jpni']:4d} x {r['jpnj']:4d}, "
                      f"largest oce domain: {r['iszi']:4d} x {r['iszj']:4d}")
        ni = nj = None
        if args.plot:
            ni, nj, _ = bestpartition(lloce, jperio, args.ncores, nn_hls, lxios=lxios)

    if args.plot:
        if lcheck:
            ni, nj = args.jpni, args.jpnj
        plot_decomposition(args.plot, lloce, jperio, ni, nj, cfg, lxios)
        print(' ')
        print(f'   decomposition {ni} x {nj} plotted in {args.plot}')


if __name__ == '__main__':
    main()
