#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Opals module

Author: Alessandro Sozza (CNR-ISAC)
Date: March 2025
"""

import os
import glob
import logging
import xarray as xr

def mpp_init():

    return None

def mpp_init_landprop():

    return None

def mpp_is_ocean():

    return None

def bestpartition(knbij, knbi, knbj, knbcnt):
    """

    find best partitions from best domain decomposition for NEMO

    Args:
    - knbij: total number if subdomains (knbi*knbj)
    - knbi, knbj: number if subdomains along i and j (knbi and knbj)
    - knbcnt: number of land subdomains
    
    """

    # get the proportion of land point over the global domain
    mpp_init_landprop(zpropland) 
    inbij = NINT( REAL(knbij, wp) / ( 1.0 - zpropland ) )

    # define the largest possible value for jpni*jpnj
    inbijmax = inbij*2

    inbimax = 0
    inbjmax = 0
    isziref = Ni0glo*Nj0glo+1
    iszjref = Ni0glo*Nj0glo+1    

    # get the list of knbj that gives a smaller jpjmax than knbj-1
    
    
    return None

def read_bathymetry():

    return None

