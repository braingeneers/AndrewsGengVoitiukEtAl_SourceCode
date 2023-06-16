# hippocampus.py
#
# Replicate the dentate gyrus network of Buchin et al. 2023, but using NEST
# and Izhikevich neurons for quick and easy simulation.
import os
os.environ['PYNEST_QUIET'] = '1'
import nest
import numpy as np
import braingeneers.analysis as ba
import matplotlib.pyplot as plt
from collections import namedtuple

plt.ion()
nest.set_verbosity('M_WARNING')
from pynestml.frontend.pynestml_frontend import generate_nest_target
generate_nest_target('models/', '/tmp/nestml-hippocampus/',
                     module_name='hippocampusmodule',
                     logging_level='WARNING')
nest.Install('hippocampusmodule')


# %%


def reset_nest(dt, seed):
    nest.ResetKernel()
    nest.local_num_threads = 12
    nest.resolution = dt
    nest.rng_seed = seed
    nest.CopyModel('izh_cond_exp2syn', 'granule_cell',
                   params=dict(C_m=65.0, tau1_exc=1.5, tau2_exc=5.5,
                               tau1_inh=0.26, tau2_inh=1.8))
    nest.CopyModel('izh_cond_exp2syn', 'basket_cell',
                   params=dict(C_m=150.0, tau1_exc=0.3, tau2_exc=0.6,
                               tau1_inh=0.16, tau2_inh=1.8))


Weights = namedtuple('Weights', 'EE EI II IE XE FE XI FI')
default_weights = Weights(
    EE=0.13,
    EI=4.7,
    II=7.6,
    IE=1.6,
    XE=3.3,
    FE=1.0,
    XI=1.5,
    FI=1.0)


def scaled_weights(factor, w=default_weights):
    '''
    Create a rescaled version of the weights w. There are four variants:
     1. For scalar `factor`, multiply all the weights by this value
     2. For tuple `(fe, fi)`, multiply the excitatory weights by `fe` and
        inhibitory weights by `fi`.
     3. For vector `factor`, multiply each weight by the corresponding
        element of the vector.
     4. For None, generate a vector of 8 random numbers from an exponential
        distribution.
    '''
    match factor:
        case None | 'random':
            x = np.random.exponential(size=8)
        case [e, i]:
            x = np.array([e, e, i, i, e, e, e, e])
        case _:
            x = np.asarray(factor)
    return Weights(*x*w)


def create_dentate_gyrus(N_granule:int=500, N_basket:int=6,
                         N_perforant:int=50,
                         w=default_weights):
    '''
    Create a dentate gyrus network for a NEST simulation, consisting of
    N_granule granule cells and N_basket basket cells, based on the dentate
    gyrus network of Buchin et al. 2023.

    The network consists of granule cells and basket cells, but they have
    been simplified from the original paper; instead of using a set of
    explicitly optimized neuron parameters, the neurons are Izhikevich
    neurons with the same parameter distributions as the excitatory and
    inhibitory cell types from the 2003 paper that introduced the model.

    Granule cells and basket cells are arranged in concentric half-circular
    arcs of radius 800μm and 750μm, and connectivity is all local. GCs
    connect to 50 of the 100 closest GCs as well as the 3 closest BCs. BCs
    connect to 100/140 closest GCs as well as their own two closest
    neighbors. There are also Poisson inputs to each neuron.

    Instead of randomizing the number of synapses, we just use uniformly
    distributed weights, equal to a number of synapses times the weight per
    synapse from the original paper.

    The original network used a distinction between dendritic and somatic
    synapses, but this is not implemented here. Instead, the synapses use
    different amplitudes and time constants to emulate the effect of
    '''
    r_g, r_b = 800, 750

    theta_g = np.linspace(0, np.pi, N_granule)
    pos_g = nest.spatial.free(
        list(zip(r_g*np.sin(theta_g), r_g*np.cos(theta_g))))
    granule = nest.Create('granule_cell', positions=pos_g)
    variate = np.random.uniform(size=N_granule)**2
    granule.c = -65 + 15*variate
    granule.d = 8 - 6*variate
    granule.V_m = -70 + 5*variate

    theta_b = np.linspace(0, np.pi, N_basket)
    pos_b = nest.spatial.free(
        list(zip(r_b*np.sin(theta_b), r_b*np.cos(theta_b))))
    basket = nest.Create('basket_cell', positions=pos_b)
    variate = np.random.uniform(size=N_basket)
    basket.a = 0.02 + 0.08*variate
    basket.b = 0.25 - 0.05*variate
    basket.V_m = -70 + 5*variate

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
                 dict(synapse_model='static_synapse', delay=0.8,
                      weight=w.EE * nest.random.uniform(2, 5)))

    # Likewise for the BCs, but instead of including a fixed number of
    # neighbors, the radius is fixed to capture one neighbor in the original
    # formulation with only 6 BCs.
    nest.Connect(basket, basket,
                 dict(rule='pairwise_bernoulli', p=1.0,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_b, 6, 1.1))),
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
        nest.Connect(b, neighbors,
                     dict(rule='fixed_outdegree', outdegree=100),
                     dict(synapse_model='static_synapse', delay=0.8,
                          weight=-nest.random.uniform(2*w.IE, 5*w.IE)))

    for g, θ in zip(granule, theta_g):
        θb = np.clip(θ, theta_b[1], theta_b[-2])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_b, 6, 1.5)))
        neighbors = nest.SelectNodesByMask(
            basket, [r_b*np.sin(θb), r_b*np.cos(θb)], mask)
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

    # The focal input is required to give the simulation a kick. It comes
    # through the "perforant path", which is supposed to trigger one lamella
    # of the hippocampus at a time, in this case just N_perforant adjacent
    # cells from the middle of the granule cell layer.
    if N_perforant > 0:
        focal = nest.Create('poisson_generator',
                            params=dict(rate=100.0, start=100.0, stop=200.0))
        focal_granule = nest.Create('parrot_neuron', 200)
        nest.Connect(focal, focal_granule, 'all_to_all')
        n = N_perforant//2
        nest.Connect(focal_granule, granule[N_granule//2-n:N_granule//2+n],
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


sd = sim(N_granule=1000, N_basket=20, T=1e4, N_perforant=0)
idces, times = sd.idces_times()
print(f'FR = {sd.rates("Hz").mean():.2f} Hz')
plt.plot(times, idces, '.', ms=1)
