# Adversarially-learned Inference via an Ensemble of Discrete Undirected Graphical Models

Undirected graphical models are compact representations of joint probability distributions over random variables. To solve inference tasks of interest, graphical models of arbitrary topology can be trained using empirical risk minimization. However, to solve inference tasks that were not seen during training, these models (EGMs) often need to be re-trained. Instead, we propose an inference-agnostic adversarial training framework which produces an infinitely-large ensemble of graphical models (AGMs). The ensemble is optimized to generate data within the GAN framework, and inference is performed using a finite subset of these models. AGMs perform comparably with EGMs on inference tasks that the latter were specifically optimized for. Most importantly, AGMs show significantly better generalization to unseen inference tasks compared to EGMs, as well as deep neural architectures like GibbsNet and VAEAC which allow arbitrary conditioning. Finally, AGMs allow fast data sampling, competitive with Gibbs sampling from EGMs.

________

The code folder contains individual files, needed for the experiments, but the main code file of interest is the file: 'README + Code Entry Point (Run experiments from this notebook)'.

That file is a jupyter notebook with exact instructions on how to re-run our experiments, and code cells provided ready to be run. 
Everything is organized such that the user only has to change the name of the dataset chosen and method chosen to run any experiment.
