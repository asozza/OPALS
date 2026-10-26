#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Regression tests for opals

Run with:
    python -m unittest -v test_opals
    python -m pytest test_opals.py

The land-sea masks are stored as small .npz files in data/ (see opals.py --save-mask),
since domain_cfg.nc files are too large for the repository.
data/orca2_ref.txt is the list printed by nemo.exe with ln_listonly = .true. (ORCA2, 256 cores).
"""

import io
import os
import tempfile
import unittest
import contextlib

import numpy as np

import opals

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')


def load(name):
    return opals.read_bathymetry(os.path.join(DATA, name))


class TestBasesplit(unittest.TestCase):

    def test_sizes_cover_global_domain(self):
        for jperio in (0, 1, 4, 6):
            for ni, nj in [(1, 1), (7, 5), (20, 15), (45, 37)]:
                for khls in (0, 1):
                    _, _, iimppt, ijmppt, ijpi, ijpj = opals.mpp_basesplit(180 + 2*khls, 148 + 2*khls, khls, ni, nj, jperio)
                    self.assertEqual(ijpi.sum() - 2*khls*(ni-1), 180 + 2*khls)
                    self.assertEqual(ijpj.sum() - 2*khls*(nj-1), 148 + 2*khls)
                    self.assertTrue((np.diff(iimppt) == ijpi[:-1] - 2*khls).all())
                    self.assertTrue((np.diff(ijmppt) == ijpj[:-1] - 2*khls).all())


class TestORCA2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lloce, cls.jperio, _ = load('orca2_mask.npz')

    def test_mask(self):
        self.assertEqual(self.lloce.shape, (148, 180))
        self.assertEqual(self.jperio, 4)

    def test_list_matches_nemo(self):
        res = opals.bestpartition(self.lloce, self.jperio, 256, ldlist=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            opals.print_list(res)
        got = [l.rstrip() for l in buf.getvalue().splitlines() if l.startswith('nb_cores')]
        with open(os.path.join(DATA, 'orca2_ref.txt')) as f:
            ref = [l.rstrip() for l in f]
        self.assertEqual(got, ref)

    def test_best_is_consistent_with_list(self):
        res = opals.bestpartition(self.lloce, self.jperio, 256, ldlist=True)
        for n in (256, 130, 100, 31):
            ni, nj, icnt = opals.bestpartition(self.lloce, self.jperio, n)
            first = next(r for r in res if r['nb_cores'] <= n)
            self.assertEqual(ni*nj - icnt, first['nb_cores'])

    def test_decomposition_info(self):
        r = opals.decomposition_info(self.lloce, self.jperio, 20, 15)
        self.assertEqual((r['nb_cores'], r['land']), (254, 46))
        self.assertEqual((r['jpimax'], r['jpjmax']), (11, 12))

    def test_best_per_node(self):
        res = opals.bestpartition(self.lloce, self.jperio, 256, ldlist=True)
        out = opals.best_per_node(res, 128)
        self.assertEqual([r['nodes'] for r in out], [1, 2, 3, 4, 5])
        self.assertEqual(out[1]['nb_cores'], 254)
        for r in out:
            self.assertTrue((r['nodes']-1)*128 < r['nb_cores'] <= r['nodes']*128)

    def test_best_per_node_extra_cores(self):
        res = opals.bestpartition(self.lloce, self.jperio, 256, ldlist=True)
        out = opals.best_per_node(res, 128, nextra=40)
        self.assertEqual(out[0]['nb_cores'], 88)                  # 88 + 40 = 128: one full node
        for r in out:
            self.assertTrue((r['nodes']-1)*128 < r['nb_cores'] + 40 <= r['nodes']*128)
            self.assertEqual(r['idle'], r['nodes']*128 - r['nb_cores'] - 40)

    def test_load_balance(self):
        levels = opals.read_levels(os.path.join(DATA, 'orca2_mask.npz'))
        self.assertEqual(levels.shape, self.lloce.shape)
        self.assertTrue(((levels > 0) == self.lloce).all())
        # one subdomain: all the wet points, no imbalance
        b2 = opals.load_balance(self.lloce, self.jperio, 1, 1)
        self.assertEqual(b2['wmax'], np.count_nonzero(self.lloce))
        self.assertAlmostEqual(b2['imbalance'], 1.0)
        b3 = opals.load_balance(self.lloce, self.jperio, 1, 1, levels=levels)
        self.assertEqual(b3['wmax'], levels.sum())
        # the inner domains of the ocean subdomains contain all the wet points
        for ni, nj in [(20, 15), (16, 16), (7, 3)]:
            b = opals.load_balance(self.lloce, self.jperio, ni, nj, levels=levels)
            nb = np.count_nonzero(opals.mpp_is_ocean(self.lloce, self.jperio, ni, nj))
            self.assertAlmostEqual(b['wmean'] * nb, levels.sum())
            self.assertGreaterEqual(b['imbalance'], 1.0)
            self.assertTrue(0. < b['wet_frac'] <= 100.)

    def test_plot(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest('matplotlib not available')
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, 'map.png')
            opals.plot_decomposition(fname, self.lloce, self.jperio, 20, 15, 'ORCA2')
            self.assertGreater(os.path.getsize(fname), 0)


class TestEORCA1(unittest.TestCase):
    """ F-pivot north fold (jperio = 6): regression on the current results """

    @classmethod
    def setUpClass(cls):
        cls.lloce, cls.jperio, _ = load('eorca1_mask.npz')

    def test_best(self):
        self.assertEqual(self.jperio, 6)
        self.assertEqual(opals.bestpartition(self.lloce, self.jperio, 256), (24, 14, 80))
        self.assertEqual(opals.bestpartition(self.lloce, self.jperio, 1024), (40, 37, 462))

    def test_list_is_monotonic(self):
        res = opals.bestpartition(self.lloce, self.jperio, 1024, ldlist=True)
        self.assertEqual(res[-1]['nb_cores'], 1)
        self.assertEqual((res[-1]['iszi'], res[-1]['iszj']), (362, 333))
        self.assertTrue(all(a['nb_cores'] > b['nb_cores'] and a['size'] < b['size'] for a, b in zip(res, res[1:])))


class TestMaskIO(unittest.TestCase):

    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        lloce = rng.random((13, 21)) > 0.4
        with tempfile.TemporaryDirectory() as tmp:
            fname = os.path.join(tmp, 'mask.npz')
            opals.save_mask(fname, lloce, 4, 'TEST')
            mask, jperio, cfg = opals.read_bathymetry(fname)
        self.assertTrue((mask == lloce).all())
        self.assertEqual((jperio, cfg), (4, 'TEST'))


if __name__ == '__main__':
    unittest.main()
