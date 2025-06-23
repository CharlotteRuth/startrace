import pynbody as pb
import numpy as np
import os
import re
import glob

def clean_simbase(filename):
    """Makes sure simbase is not a step by removing steps from the name"""
    new_string = ''.join([m + '/' for m in filename.split('/') if '.0' not in m])
    return new_string

def list_steps(sim_base, pattern=r'\.(0{2}\d{4})$'):
    """Given the base directory of the simulation, return full paths of all steps"""
    
    sim_base = clean_simbase(sim_base)
    files = os.listdir(sim_base)
    pattern = pattern
    steplist = []
    for fil in files:
        m = re.search(pattern, fil)
        if m is not None:
            steplist.append(m.string)
            
    steplist = np.sort(steplist)[::-1]
    pathlist = np.array(list(filter(None, [step_path(sim_base, step) for step in steplist])), dtype='object')
    #pathlist
    
    return pathlist

def step_path(sim_base, step):
    """Given the base directory of the simulation and the step, return the path"""
    
    try:
        a = int(step)
        step = str(step).rjust(6, '0')
        step = max(sim_base.split('/'), key=len) + '.' + step
    except ValueError:
        pass
    
    #if sim_base[-1] != '/':
    #    sim_base += '/'
    sim_base = clean_simbase(sim_base)
    try:
        test = pb.load(sim_base + step)
    except OSError:
        try:
            test = pb.load(sim_base + step + '/' + step)
        except OSError:
            print(step + ' does not exist or has incorrect permissions')
            return None
    filename = test.filename
    del test
    return filename

