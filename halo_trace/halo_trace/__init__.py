from . import paths
from . import tracing
from . import amiga_ahf
from . import satfinding

import pynbody as pb
import numpy as np
import os
import pandas as pd
import glob


#def match_catalogs(cat_1_to_2, sim1_grp_list, threshold=0.001, verbose=0):
#    """Find the most likely halo matches between two simulations
#    
#    cat - a fuzzy_match_catalog object of two sims
#    grp_list - the list of grps of sim1, for which to find matches in sim 2
#    threshold (default 0.5%) - minimum fraction of particles in common for a match,
#    used after eliminating previously matched halos
#    
#    If the most likely match was already used, it will go to the next most likely. If all
#    are previously used, defaults to the most in common
#    
#    returns - sim 2 grp list, with -1 for no matches
#    """
#    
#    sim2_from_sim1 = np.zeros(len(sim1_grp_list), dtype=int)
#    for i, grp in enumerate(sim1_grp_list):
#        if len(cat_1_to_2[grp]) == 0:
#            if verbose > 0:
#                print(grp, 'No match! Nothing in common.')
#            sim2_from_sim1[i] = -1
#        else:
#            good = False
#            if verbose > 2:
#                print(grp, cat_1_to_2[grp])
#            for j in range(len(cat_1_to_2[grp])):
#                if not good:
#                    if (cat_1_to_2[grp][j][0] not in sim2_from_sim1
#                        and cat_1_to_2[grp][j][1] > threshold):
#                        sim2_from_sim1[i] = cat_1_to_2[grp][j][0]
#                        good = True
#
#            if not good:
#                if verbose > 0:
#                    print(grp, 'Overlap with other choices. Defaulting to highest (non-main halo) overlap.')
#                if verbose > 1:
#                    print(grp, cat_1_to_2[grp])
#                if cat_1_to_2[grp][0][0] == 1 and len(cat_1_to_2[grp]) > 1:
#                    sim2_from_sim1[i] = cat_1_to_2[grp][1][0]
#                else:
#                    sim2_from_sim1[i] = cat_1_to_2[grp][0][0]
#    
#    return sim2_from_sim1
#
