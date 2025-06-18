from base import *
import os

# function to log uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    global logger
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception

hubble =  0.6776942783267969

def run_tracking(sim, z0haloid, filepaths,haloids,h1ids):
    output = pd.DataFrame()
    logger.debug('Starting run')
    for f,haloid,h1id in zip(filepaths,haloids,h1ids):
        s = pynbody.load(f)
        s.physical_units()
        h = s.halos()
        halo = h[haloid]
        h1 = h[h1id]
        snapnum = f[-4:]
        logger.debug(f'* Snapshot {snapnum}')
        output = pd.concat([output, analysis(s,halo,h1,h,haloid,h1id)])
    
    return output


def analysis(s,halo,h1,h,haloid,h1id):
    output = pd.DataFrame()
    a = float(s.properties['a'])

    # calculate properties that are invariant to centering
    output['time'] = [float(s.properties['time'].in_units('Gyr'))]
    
    # calculate properties centered on the satellite
    pynbody.analysis.halo.center(halo)
    Rvir = halo.properties['Rvir'] * a / hubble
    bins = np.power(10, np.arange(0, np.log10(Rvir), 0.02))
    p = pynbody.analysis.profile.Profile(halo,bins=bins,ndim=3)
    output['bins'] = [bins]
    output['mass_enc'] = [np.array(p['mass_enc'])]
    
    return output




if __name__ == '__main__':
    sim = str(sys.argv[1])
    z0haloid = int(sys.argv[2])

    if not os.path.exists('./logs/'):
        os.mkdir('./logs/')
    logging.basicConfig(filename=f'./logs/{sim}_{z0haloid}.log', 
                        format='%(asctime)s :: %(name)s :: %(levelname)-8s :: %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)
    logger = logging.getLogger('PartTracker')
    logger = logging.getLogger('SatProfiles')

    logger.debug(f'--------------------------------------------------------------')
    logger.debug(f'Beginning satellite profile computation for {sim}-{z0haloid}')
    # in order: debug, info, warning, error

    logger.debug('Getting stored filepaths and haloids')
    filepaths, haloids, h1ids = get_stored_filepaths_haloids(sim,z0haloid)
    # filepaths starts with z=0 and goes to z=15 or so

    logger.debug('Getting starting snapshot (may take a while)')
    snap_start = get_snap_start(sim,z0haloid)
    logger.debug(f'Start on snapshot {snap_start}, {filepaths[snap_start][-4:]}')
    
    # fix the case where the satellite doesn't have merger info prior to 
    if len(haloids) < snap_start:
        snap_start = len(haloids)
        raise Exception('Careful! You may have an error since the satellite doesnt have mergertree info out to the time where you want to start. This case is untested')
    
    if len(haloids) > snap_start:
        filepaths = np.flip(filepaths[:snap_start+1])
        haloids = np.flip(haloids[:snap_start+1])
        h1ids = np.flip(h1ids[:snap_start+1])

    if len(haloids) == snap_start:
        filepaths = np.flip(filepaths[:snap_start])
        haloids = np.flip(haloids[:snap_start])
        h1ids = np.flip(h1ids[:snap_start])   
        
    # filepaths and haloids now go the "right" way, i.e. starts from start_snap and goes until z=0
    assert len(filepaths) == len(haloids)
    assert len(haloids) == len(h1ids)

    # we save the data as an .hdf5 file since this is meant for large datasets, so that should work pretty good
    output = run_tracking(sim, z0haloid, filepaths, haloids, h1ids)

    savepath = '../../Data/profiles.hdf5'
    logger.debug(f'Saving output to {savepath}')
    output.to_hdf(savepath,key=f'{sim}_{z0haloid}')







    





                
