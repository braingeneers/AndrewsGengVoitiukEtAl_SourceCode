import nest
import numpy as np
import braingeneers.analysis as ba
from collections import namedtuple


def reset_nest(dt, seed):
    nest.ResetKernel()
    nest.local_num_threads = 10
    nest.resolution = dt
    nest.rng_seed = seed

    nest.CopyModel('izhikevich', 'granule_cell')
    nest.CopyModel('izhikevich', 'basket_cell')


Weights = namedtuple('Weights', 'EE EI II IE XE FE XI FI')
default_weights = Weights(
    EE=0.13,
    EI=4.7,
    II=7.6,
    IE=1.6,
    XE=3.3,
    FE=1.0,
    XI=1.5,
    FI=1.0,
)


# Some weights that made it kind of almost work? The commented value of EE
# was very close (basic background activity) but without the big seizures,
# so I increased it by 0.5 and there are strong traveling seizures.
test_weights = Weights(
    # EE=0.3252315790014848,
    EE=0.8252315790014848,
    EI=0.06144751886110525,
    II=12.30122120344486,
    IE=2.043460990424841,
    XE=7.035798517061904,
    FE=0.15162233054392354,
    XI=0.5029318414532769,
    FI=0.8824668381047678)


def random_weights():
    return Weights(*np.random.exponential(size=8) * default_weights)


def scaled_weights(factor):
    return Weights(*np.multiply(factor, default_weights))


def create_dentate_gyrus(N_granule:int=500, N_basket:int=6,
                         w=default_weights):
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
    r_g, r_b = 800, 750

    theta_g = np.linspace(0, np.pi, N_granule)
    pos_g = nest.spatial.free(
        list(zip(r_g*np.sin(theta_g), r_g*np.cos(theta_g))))
    granule = nest.Create('granule_cell', positions=pos_g)

    theta_b = np.linspace(0, np.pi, N_basket)
    pos_b = nest.spatial.free(
        list(zip(r_b*np.sin(theta_b), r_b*np.cos(theta_b))))
    basket = nest.Create('basket_cell', positions=pos_b)

    def r_kth_nearest(radius, N, k):
        'The distance to the kth nearest neighbor on a half-circle.'
        angle_per_step = np.pi/(N-1)
        return 2*radius * np.sin(k*angle_per_step/2)

    # Connect the granule cells to each other with a circular mask that only
    # grabs the 100 nearest neighbors, and a fixed degree of 50. Note that
    # this means going for the 50th-nearest neighbor, as the radius extends
    # in both directions.
    # AMPA_ExcToExc_GC_GC
    nest.Connect(granule, granule,
                 dict(rule='fixed_outdegree', outdegree=50,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_g, N_granule, 50))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse', delay=0.8,
                      weight=w.EE * nest.random.uniform(2, 5)))

    # Likewise for the BCs, but they only connect to immediate neighbors.
    # GABA_InhToInh_BC_BC
    nest.Connect(basket, basket,
                 dict(rule='pairwise_bernoulli', p=1.0,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_b, N_basket, 1.5))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse', delay=0.8,
                      weight=-w.II * nest.random.uniform(2, 5)))

    # For between-population connections, find the nearest point in the
    # other population by calculating the position of the nearest neuron in
    # the other layer and using that as the anchor for the mask.
    # GABA_InhToExc_BC_GC
    for b, θ in zip(basket, theta_b):
        θg = np.clip(θ, theta_g[69], theta_g[-70])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_g, N_granule, 70)))
        neighbors = nest.SelectNodesByMask(
            granule, [r_g*np.sin(θg), r_g*np.cos(θg)], mask)
        assert len(neighbors) == 140
        nest.Connect(b, neighbors,
                     dict(rule='fixed_outdegree', outdegree=100),
                     dict(synapse_model='static_synapse', delay=0.8,
                          weight=-nest.random.uniform(2*w.IE, 5*w.IE)))

    # AMPA_ExctoInh_GC_BC
    for g, θ in zip(granule, theta_g):
        θb = np.clip(θ, theta_b[1], theta_b[-2])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_b, N_basket, 1.5)))
        neighbors = nest.SelectNodesByMask(
            basket, [r_b*np.sin(θb), r_b*np.cos(θb)], mask)
        assert len(neighbors) == 3
        nest.Connect(g, neighbors,
                     dict(rule='pairwise_bernoulli', p=1.0),
                     dict(synapse_model='static_synapse', delay=0.8,
                          weight=nest.random.uniform(2*w.EI, 5*w.EI)))

    # Finally create the Poisson inputs to all of this...
    noise = nest.Create('poisson_generator', params=dict(rate=15.0))
    for layer, wX in ((granule, w.XE), (basket, w.XI)):
        nest.Connect(noise, layer, 'all_to_all',
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(5*wX, 15*wX)))

    # The focal input is required to give the simulation a kick.
    focal = nest.Create('poisson_generator',
                        params=dict(rate=100.0, start=100.0, stop=200.0))
    focal_granule = nest.Create('parrot_neuron', 200)
    nest.Connect(focal, focal_granule, 'all_to_all')
    nest.Connect(focal_granule, granule,
                 dict(rule='fixed_indegree', indegree=100),
                 dict(synapse_model='static_synapse',
                      weight=nest.random.uniform(5*w.FE, 15*w.FE)))
    nest.Connect(focal, basket, 'all_to_all',
                 dict(synapse_model='static_synapse',
                      weight=nest.random.uniform(10*w.FI, 30*w.FI)))

    return granule, basket


def sim(T=1e3, dt=0.1, seed=42, **kwargs):
    reset_nest(dt, seed)
    granule, basket = create_dentate_gyrus(**kwargs)
    rec = nest.Create('spike_recorder')
    nest.Connect(granule, rec)
    nest.Connect(basket, rec)
    nest.Simulate(T)
    # This is a little weird, but I want to use the spike train extraction
    # code I wrote for SpikeData, but NEST NodeCollections can't be combined
    # once they have spatial metadata etc. Instead, create a SpikeData per
    # population, and combine their actual data.
    sdg, sdb = [
        ba.SpikeData(rec, layer, N=len(layer), length=T)
        for layer in (granule, basket)]
    return ba.SpikeData(sdg.train + sdb.train, length=T)

sd = sim(w=test_weights)
print(f'FR = {sd.rates("Hz").mean()}')
idces, times = sd.idces_times()
plt.figure()
plt.plot(*sd.idces_times()[::-1], ',')

# ws = []
# for i in range(20):
#     # ws.append(scaled_weights(0.2*(i+1)))
#     ws.append(random_weights())
#     sd = sim(seed=42, w=ws[-1])
#     print(f'Culture {i} FR = {sd.rates("Hz").mean()}')
#     idces, times = sd.idces_times()
#     if max(times) > 250:
#         plt.figure()
#         plt.title(f'Culture {i}')
#         plt.plot(*sd.idces_times()[::-1], ',')
