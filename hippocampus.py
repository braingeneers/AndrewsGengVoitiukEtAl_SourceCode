import nest
import numpy as np
import braingeneers.analysis as ba


def reset_nest(dt, seed):
    nest.ResetKernel()
    nest.local_num_threads = 10
    nest.resolution = dt
    nest.rng_seed = seed



def create_dentate_gyrus(N:int):
    '''
    Create a dentate gyrus network for a NEST simulation, consisting of
    N granule cells and N basket cells, based on the dentate gyrus network
    of Buchin et al. 2023.

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
    theta = np.linspace(0, np.pi, N)
    r_granule, r_basket = 1500, 750
    pos_granule = nest.spatial.free(
        list(zip(r_granule*np.sin(theta), r_granule*np.cos(theta))))
    pos_basket = nest.spatial.free(
        list(zip(r_basket*np.sin(theta), r_basket*np.cos(theta))))
    granule = nest.Create('iaf_cond_alpha', positions=pos_granule)
    basket = nest.Create('iaf_cond_alpha', positions=pos_basket)

    def r_kth_nearest(radius, k):
        'The distance to the kth nearest neighbor on a half-circle.'
        return radius*np.pi*k/(N-1)

    # Connect the granule cells to each other with a circular mask that only
    # grabs the 100 nearest neighbors, and a fixed degree of 50. Note that
    # this means going for the 50th-nearest neighbor, as the radius extends
    # in both directions.
    w = 5.0
    nest.Connect(granule, granule,
                 dict(rule='fixed_outdegree', outdegree=50,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_granule, 50))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse',
                      weight=nest.random.uniform(2*w, 5*w)))

    # Likewise for the BCs, but they only connect to immediate neighbors.
    nest.Connect(basket, basket,
                 dict(rule='pairwise_bernoulli', p=1.0,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_basket, 1))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse',
                      weight=nest.random.uniform(2*w, 5*w)))

    # For between-population connections, find the nearest point in the
    # other population by calculating the position of the nearest neuron in
    # the other layer and using that as the anchor for the mask.
    # Unfortunately, the anchor can only be a number, not a parameter, so
    # this has to be done in a loop.
    for b, θ in zip(basket, theta):
        θ = np.clip(θ, theta[69], theta[-70])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_granule, 70)))
        neighbors = nest.SelectNodesByMask(
            granule, [r_granule*np.sin(θ), r_granule*np.cos(θ)], mask)
        nest.Connect(b, neighbors,
                     dict(rule='fixed_outdegree', outdegree=100),
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(2*w, 5*w)))

    for g, θ in zip(granule, theta):
        θ = np.clip(θ, theta[1], theta[-2])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_basket, 1)))
        neighbors = nest.SelectNodesByMask(
            basket, [r_basket*np.sin(θ), r_basket*np.cos(θ)], mask)
        assert len(neighbors) == 3
        nest.Connect(g, neighbors,
                     dict(rule='pairwise_bernoulli', p=1.0),
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(2*w, 5*w)))

    # Finally create the Poisson inputs to all of this...
    noise = nest.Create('poisson_generator', params=dict(rate=15.0))
    for layer in (granule, basket):
        nest.Connect(noise, layer, 'all_to_all',
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(5*w, 15*w)))

    return granule, basket


def sim():
    reset_nest(0.1, 42)
    granule, basket = create_dentate_gyrus(506)
    rec = nest.Create('spike_recorder')
    nest.Connect(granule, rec)
    nest.Connect(basket, rec)
    nest.Simulate(10e3)
    return [
        ba.SpikeData(rec, layer, length=10e3, N=506)
        for layer in (granule, basket)]

sds = sim()
