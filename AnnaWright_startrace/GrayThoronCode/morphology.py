import numpy as np
from sklearn.neighbors import NearestNeighbors
import astropy.units as u


def calc_w_v_nat(pos, vel):
    # w_v methodology from 2021ApJ...920...10P
    assert pos.shape == vel.shape
    assert pos.ndim == 2
    assert pos.shape[-1] == 3
    sigma_pos = np.linalg.norm(np.std(pos, axis=0))
    sigma_vel = np.linalg.norm(np.std(vel, axis=0))
    return (sigma_pos / sigma_vel)


def local_velocity_dispersion(pos, vel, w_v=None, k=None, return_disps=False, max_chunk_gb=10, n_jobs=None, test_indices=None):
    # if not specified, use 2021ApJ...920...10P methodology
    if w_v is None:
        w_v = calc_w_v_nat(pos, vel).to(u.kpc / (u.km / u.s))
    if k is None:
        k = 20 if len(pos) > 300 else 7

    if test_indices is None:
        # use all particles
        test_indices = np.arange(len(pos))
    Npart = len(test_indices)

    # chunk size based on how large Nneighbors x chunk array can be
    chunk = int(max_chunk_gb*10**9 / (8 * k))
    assert chunk > 0

    # don't parallelize on too small of a query: https://github.com/scikit-learn/scikit-learn/issues/6645
    n_jobs_used = n_jobs
    if ((chunk < 10**4) or (Npart < 10**4)) and not (n_jobs is None):
        n_jobs_used = None

    # construct tree
    w = np.hstack([pos, vel * w_v]).to(u.kpc)
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=n_jobs_used).fit(w)

    # print off mode of calculation
    mode = 'serial' if (n_jobs_used is None) else f'parallel (n_jobs = {n_jobs_used})'
    endmsg = 'without chunking' if chunk > Npart else f'with chunksize {chunk}'
    print(f"Computing in {mode} {endmsg}")

    # use code as if you're chunking, but if chunk > Npart it'll run just as fast
    start = 0
    dispersions = np.array([])
    while start < Npart:
        end = start + chunk
        if end > Npart:
            end = Npart

        neighbor_indices = nbrs.kneighbors(w[test_indices[start:end]])[1]
        dispersion_chunk = np.linalg.norm(np.std(vel[neighbor_indices], axis=1), axis=1)
        dispersions = np.append(dispersions, dispersion_chunk.to(u.km/u.s).value)
        start += chunk

    stats = np.percentile(dispersions, [16, 50, 84])

    return [stats, dispersions, test_indices] if return_disps else stats