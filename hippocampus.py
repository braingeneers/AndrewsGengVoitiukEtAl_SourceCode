import nest
import numpy as np
import braingeneers.analysis as ba
from dataclasses import dataclass


def reset_nest(dt, seed):
    nest.ResetKernel()
    nest.local_num_threads = 10
    nest.resolution = dt
    nest.rng_seed = seed


@dataclass(kw_only=True, unsafe_hash=True)
class Weights:
    EE: float = 0.00013e4
    EI: float = 0.0047e4
    II: float = 0.0076e4
    IE: float = 0.0016e4
    XE: float = 0.0003e4
    FE: float = 0.001e4
    XI: float = 0.0005e4
    FI: float = 0.001e4


def create_dentate_gyrus(N_granule:int, N_basket:int, w:Weights=Weights()):
    '''
    Create a dentate gyrus network for a NEST simulation, consisting of
    N_granule granule cells and N_basket basket cells, based on the dentate
    gyrus network of Buchin et al. 2023.

    Granule cells and basket cells are arranged in concentric half-circular
    arcs of radius 1500μm and 750μm, and connectivity is all local. GCs
    connect to 50 of the 100 closest GCs as well as the 3 closest BCs. BCs
    connect to 100/140 closest GCs as well as their own two closest
    neighbors. There are also Poisson inputs to each neuron.

    Instead of randomizing the number of synapses, we just use uniformly
    distributed weights.

    The original network used a distinction between dendritic and somatic
    synapses, but this is not implemented here. Instead, the synapses use
    different amplitudes and time constants to emulate the effect of
    '''
    # Create 2N nodes at positions along the arcs.
    theta_g = np.linspace(0, np.pi, N_granule)
    theta_b = np.linspace(0, np.pi, N_basket)
    r_g, r_b = 800, 750
    pos_g = nest.spatial.free(
        list(zip(r_g*np.sin(theta_g), r_g*np.cos(theta_g))))
    pos_b = nest.spatial.free(
        list(zip(r_b*np.sin(theta_b), r_b*np.cos(theta_b))))
    granule = nest.Create('iaf_cond_alpha', positions=pos_g)
    basket = nest.Create('iaf_cond_alpha', positions=pos_b)

    def r_kth_nearest(radius, N, k):
        'The distance to the kth nearest neighbor on a half-circle.'
        angle_per_step = np.pi/(N-1)
        return 2*radius * np.sin(k*angle_per_step/2)

    # Connect the granule cells to each other with a circular mask that only
    # grabs the 100 nearest neighbors, and a fixed degree of 50. Note that
    # this means going for the 50th-nearest neighbor, as the radius extends
    # in both directions.
    nest.Connect(granule, granule,
                 dict(rule='fixed_outdegree', outdegree=50,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_g, N_granule, 50))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse',
                      weight=nest.random.uniform(2*w.EE, 5*w.EE)))

    # Likewise for the BCs, but they only connect to immediate neighbors.
    nest.Connect(basket, basket,
                 dict(rule='pairwise_bernoulli', p=1.0,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_b, N_basket, 1))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse',
                      weight=nest.random.uniform(2*w.II, 5*w.II)))

    # For between-population connections, find the nearest point in the
    # other population by calculating the position of the nearest neuron in
    # the other layer and using that as the anchor for the mask.
    # Unfortunately, the anchor can only be a number, not a parameter, so
    # this has to be done in a loop.
    for b, θ in zip(basket, theta_b):
        θ = np.clip(θ, theta_g[69], theta_g[-70])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_g, N_granule, 70)))
        neighbors = nest.SelectNodesByMask(
            granule, [r_g*np.sin(θ), r_g*np.cos(θ)], mask)
        nest.Connect(b, neighbors,
                     dict(rule='fixed_outdegree', outdegree=100),
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(2*w.IE, 5*w.IE)))

    for g, θ in zip(granule, theta_g):
        θ = np.clip(θ, theta_b[1], theta_b[-2])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_b, N_basket, 1.5)))
        neighbors = nest.SelectNodesByMask(
            basket, [r_b*np.sin(θ), r_b*np.cos(θ)], mask)
        nest.Connect(g, neighbors,
                     dict(rule='pairwise_bernoulli', p=1.0),
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(2*w.EI, 5*w.EI)))

    # Finally create the Poisson inputs to all of this...
    noise = nest.Create('poisson_generator', params=dict(rate=15.0))
    for layer, wX in ((granule, w.XE), (basket, w.XI)):
        nest.Connect(noise, layer, 'all_to_all',
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(5*wX, 15*wX)))

    return granule, basket


def sim(T=10e3, N_granule=500, N_basket=6, dt=0.1, seed=42):
    reset_nest(dt, seed)
    granule, basket = create_dentate_gyrus(N_granule, N_basket)
    rec = nest.Create('spike_recorder')
    nest.Connect(granule, rec)
    nest.Connect(basket, rec)
    nest.Simulate(T)
    # This is a little weird, but I want to use the spike train extraction
    # code I wrote for SpikeData, but NEST NodeCollections can't be combined
    # once they have spatial metadata etc. Instead, create a SpikeData per
    # population, and combine their actual data.
    sdg, sdb = [
        ba.SpikeData(rec, layer, N=len(layer))
        for layer in (granule, basket)]
    return ba.SpikeData(sdg.train + sdb.train, length=10e3)

sd = sim()
