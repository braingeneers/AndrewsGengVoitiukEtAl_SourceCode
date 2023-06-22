# hippocampus.py
#
# Replicate the dentate gyrus network of Buchin et al. 2023, but using
# NEST and Izhikevich neurons for quick and easy simulation.
import os
os.environ['PYNEST_QUIET'] = '1'
import nest
import numpy as np
import braingeneers.analysis as ba
import matplotlib.pyplot as plt
from collections import namedtuple
from tqdm import tqdm

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
    np.random.seed(seed)
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


def create_dentate_gyrus(N_granule=500, N_basket=6, N_perforant=50,
                         p_opto=0.0, g_opto=50.0, w=default_weights):
    '''
    Create a dentate gyrus network for a NEST simulation, consisting of
    N_granule granule cells and N_basket basket cells, based on the
    dentate gyrus network of Buchin et al. 2023.

    The network consists of granule cells and basket cells, but they have
    been simplified from the original paper; instead of using a set of
    explicitly optimized neuron parameters, the neurons are Izhikevich
    neurons with the same parameter distributions as the excitatory and
    inhibitory cell types from the 2003 paper that introduced the model.

    Granule cells and basket cells are arranged in concentric
    half-circular arcs of radius 800μm and 750μm, and connectivity is all
    local. GCs connect to 50 of the 100 closest GCs as well as the
    3 closest BCs. BCs connect to 100/140 closest GCs as well as their
    own two closest neighbors. There are also Poisson inputs to each
    neuron.

    Instead of randomizing the number of synapses, we just use uniformly
    distributed weights, equal to a number of synapses times the weight
    per synapse from the original paper.
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
    granule.g_opto = g_opto * (np.random.rand(N_granule) < p_opto)

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

    # Connect the granule cells to each other with a circular mask that
    # only grabs the 100 nearest neighbors, and a fixed degree of 50.
    # Note that this means going for the 50th-nearest neighbor, as the
    # radius extends in both directions.
    nest.Connect(granule, granule,
                 dict(rule='fixed_outdegree', outdegree=50,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_g, N_granule, 50))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse', delay=1.0,
                      weight=w.EE * nest.random.uniform(2, 5)))

    # Likewise for the BCs, but instead of including a fixed number of
    # neighbors, the radius is fixed to capture one neighbor in the
    # original formulation with only 6 BCs.
    nest.Connect(basket, basket,
                 dict(rule='pairwise_bernoulli', p=1.0,
                      mask=dict(circular=dict(
                          radius=r_kth_nearest(r_b, 6, 1.1))),
                      allow_autapses=False),
                 dict(synapse_model='static_synapse', delay=1.0,
                      weight=-w.II * nest.random.uniform(2, 5)))

    # For between-population connections, find the nearest point in the
    # other population by calculating the position of the nearest neuron
    # in the other layer and using that as the anchor for the mask.
    for b, θ in zip(basket, theta_b):
        θg = np.clip(θ, theta_g[69], theta_g[-70])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_g, N_granule, 70)))
        neighbors = nest.SelectNodesByMask(
            granule, [r_g*np.sin(θg), r_g*np.cos(θg)], mask)
        nest.Connect(b, neighbors,
                     dict(rule='fixed_outdegree', outdegree=100),
                     dict(synapse_model='static_synapse', delay=1.0,
                          weight=-nest.random.uniform(2*w.IE, 5*w.IE)))

    for g, θ in zip(granule, theta_g):
        θb = np.clip(θ, theta_b[1], theta_b[-2])
        mask = nest.CreateMask(
            'circular', dict(radius=r_kth_nearest(r_b, 6, 1.5)))
        neighbors = nest.SelectNodesByMask(
            basket, [r_b*np.sin(θb), r_b*np.cos(θb)], mask)
        nest.Connect(g, neighbors,
                     dict(rule='pairwise_bernoulli', p=1.0),
                     dict(synapse_model='static_synapse', delay=1.0,
                          weight=nest.random.uniform(2*w.EI, 5*w.EI)))

    # Finally create the Poisson inputs to all of this...
    noise = nest.Create('poisson_generator', params=dict(rate=15.0))
    for layer, wX in ((granule, w.XE), (basket, w.XI)):
        nest.Connect(noise, layer, 'all_to_all',
                     dict(synapse_model='static_synapse',
                          weight=nest.random.uniform(5*wX, 15*wX)))

    # The focal input is required to give the simulation a kick. It comes
    # through the "perforant path", which is supposed to trigger one
    # lamella of the hippocampus at a time, in this case just N_perforant
    # adjacent cells from the middle of the granule cell layer.
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


def sim(T=2e3, dt=0.1, seed=42, opto_threshold=100, opto_duration=15,
        warmup_time=1e3, **kwargs):
    # Create and warm up the network.
    reset_nest(dt, seed)
    granule, basket = create_dentate_gyrus(**kwargs)
    with tqdm(total=T+warmup_time) as pbar:
        with nest.RunManager():
            for t in range(int(warmup_time)):
                nest.Run(1.0)
                pbar.update()

        # Add spike recording.
        rec = nest.Create('spike_recorder')
        nest.Connect(granule, rec)
        nest.Connect(basket, rec)

        # Simulate 1ms at a time, checking if at least opto_threshold
        # spikes occur, and enabling opto for opto_duration if so.
        opto_times = []
        has_opto = not (kwargs.get('p_opto') == 0 or opto_duration == 0
                        or opto_threshold > len(granule)+len(basket))
        with nest.RunManager():
            time_to_enable_opto = 0
            for t in range(int(T)):
                last_n_events = rec.n_events
                if time_to_enable_opto > 0:
                    time_to_enable_opto -= 1
                if has_opto:
                    granule.opto = time_to_enable_opto > 0
                nest.Run(1.0)
                pbar.update()
                new_spikes = rec.n_events - last_n_events
                if time_to_enable_opto == 0 and new_spikes > opto_threshold:
                    time_to_enable_opto = opto_duration
                    if has_opto:
                        opto_times.append(t)

    # This is a little weird, but I want to use the spike train
    # extraction code I wrote for SpikeData, but NEST NodeCollections
    # can't be combined once they have spatial metadata etc. Instead,
    # create a SpikeData per population, removing the warmup time from
    # each, and combine them.
    sdg, sdb = [
        ba.SpikeData(rec, layer, N=len(layer), length=T+warmup_time
                     ).subtime(warmup_time, ...)
        for layer in (granule, basket)]
    return ba.SpikeData(sdg.train + sdb.train, length=T,
                        metadata=dict(opto_times=opto_times,
                                      p_opto=kwargs.get('p_opto'),
                                      opto_duration=opto_duration,
                                      opto_threshold=opto_threshold))


def plot_sds(f, sds):
    '''
    Plot the raster, firing rates, and opto events for a list of spike
    rasters on the given figure.
    '''
    f.clear()
    axes = f.subplots(len(sds), 1)
    for ax, sd in zip(axes, sds):
        idces, times = sd.idces_times()
        ax.plot(times, idces, 'k|', ms=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, 2e3)
        opto_duration = sd.metadata['opto_duration']
        optos = np.array(sd.metadata['opto_times'])
        if len(optos) > 0 and opto_duration > 0:
            pc = plt.matplotlib.collections.PatchCollection(
                [plt.Rectangle((opto, -25), opto_duration, sd.N+50)
                 for opto in optos],
                facecolor='g', alpha=0.3, edgecolor='none')
            ax.add_collection(pc)
        xlim = ax.get_xlim()
        ax2 = ax.twinx()
        ax2.plot(sd.binned(1), c='purple', lw=0.75)
        ax2.set_yticks([0, 300])
        ax2.set_ylim(-25, 325)
        ax2.set_ylabel('Pop. Rate (Hz)')
    ticks = np.array([0, 10, 20])
    axes[-1].set_xticks(ticks*100, [f'${t/10:0.1f}$' for t in ticks])
    axes[-1].set_xlabel('Time (sec)')
    return axes


def query_save(f, name):
    if input('Save? [y/N] ').strip().lower().startswith('y'):
        f.savefig(name, bbox_inches='tight', dpi=600)


# %%
# Run the simulation with three different levels of optogenetic
# activation. The parameter p_opto being swept controls what fraction of
# the granule cells respond to the optogenetic feedback that is used for
# feedback. Plot two different figures of the same results, one in terms
# of rasters, and one in terms of population rate.

T_opto = 50

sds_fraction = []
for p_opto in [0.0, 0.25, 0.5, 0.75]:
    sd = sim(N_granule=1000, N_basket=12, N_perforant=0,
             p_opto=p_opto, opto_duration=T_opto, opto_threshold=100)
    idces, times = sd.idces_times()
    print(f'With {p_opto = :.0%}, '
          f'FR was {sd.rates("Hz").mean():.2f} Hz. '
          f'Did opto {len(sd.metadata["opto_times"])} times.')
    sds_fraction.append(sd)

f = plt.figure(f'Varying Optogenetic Fraction',
               figsize=(6.4, 6.4))
axes = plot_sds(f, sds_fraction)
for sd, ax in zip(sds_fraction, axes):
    ax.set_ylabel(f'$p_\\text{{opto}} = '
                  f'{100*sd.metadata["p_opto"]:.0f}\\%$')

query_save(f, f'opto-fraction-{T_opto}ms.png')
