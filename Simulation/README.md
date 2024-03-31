# Hippocampal Seizure Simulation Code

This is the simulation code corresponding to the proof of concept hippocampal model from the manuscript by Andrews & Voitiuk et al. "Optogenetic modulation of epileptiform activity in human brain tissue".


## Overview

This model was based on a previously published in silico model of the human dentate gyrus[^40] which has previously been used to study disease progression in epilepsy[^21].
Excitatory and inhibitory neurons were modeled as point neurons using a simplified dynamical model[^41] which was modified to use conductance-based synapses.
The network, increased in size from the original in order to allow seizure-like events to propagate for a longer duration, consisted of 1000 excitatory granule cells (GCs) and 12 inhibitory basket cells (BCs) evenly spaced on concentric half-circles of radius 800 µm and 750 µm respectively.
Each GC was randomly connected via AMPA synapses to 50 of the 100 nearest GCs as well as the three nearest BCs, and each BC was randomly connected via GABA synapses to 100 of the nearest 140 GCs as well as the two neighboring BCs.
All synapses use a conductance whose time course follows a difference-of-exponentials form with parameters taken from Buchin et al.[^21].
Dynamical parameters of the individual neurons were taken from Izhikevich et al.[^41], but the original model does not include membrane capacitance due to its use of instantaneous synapses, so the membrane capacitance for each cell type was selected in order to recapitulate the dynamical behavior observed in Buchin et al.[^21].

In order to interrupt the seizure-like synchronization behavior observed in this model, a light-sensitive kalium channelrhodopsin[^22] was added to the model. This channel was modeled as a simple switched conductance without dynamics, whose reversal potential and total conductance are given by the reference.
The channel was enabled in a variable fraction $p_\text{opto}$ of simulated GCs.
Then, during simulations, the optogenetic channel was controlled in closed loop: simulation was performed in 1 ms steps, and if the total spike count in a single step exceeded a threshold $x_\text{opto}$, the optogenetic channel was enabled for a duration $T_\text{opto}$.

The first second of each simulation was ignored in order to allow the network to reach its steady-state behavior.
All simulations were implemented using the NEST simulator version 3.4[^42], with models implemented using the modeling language NESTML[^43].


## How To Run

The following commands will create a Conda environment from the file `environment.yml`, activate it, and run the simulation code. You will need to have `conda` already installed, but all the specific dependencies will be included in the created environment. This will reproduce the Extended Data figure demonstrating the effect of optogenetic feedback on this simulation model:

```bash
conda env create -f environment.yml
conda activate AndrewsVoitiukEtAl
python simulation.py
```


## References

[^21]: Buchin, A. et al. Multi-modal characterization and simulation of human epileptic circuitry. Cell Reports 41, 111873 (2022).
[^22]: Govorunova, E. G. et al. Kalium channelrhodopsins are natural light-gated potassium channels that mediate optogenetic inhibition. Nature Neuroscience 25, 967-974 (2022).
[^40]: Santhakumar V. et al. Role of mossy fiber sprouting and mossy cell loss in hyperexcitability: a network model of the dentate gyrus incorporating cell types and axonal topography. J. Neurophysiol. 93, 437-453 (2005).
[^41]: Izhikevich, E. M. Simple model of spiking neurons. IEEE Transactions on Neural Networks 14, 1569-1572 (2003).
[^42]: Sinha, A. et al. NEST 3.4. Zenodo doi://10.5281/zenodo.6867800
[^43]: Linssen, Charl A.P. et al. NESTML 5.2.0. Zenodo doi://10.5281/zenodo.7648959
