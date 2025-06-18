from analysis import *

keys = get_keys()
print(keys)
print(len(keys))

for key in keys:
    sim = str(key[:4])
    haloid = int(key[5:])
    expelled, accreted = calc_ejected_expelled(sim, haloid, save=True, verbose=False)
