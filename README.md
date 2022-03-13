# Summary
**In short:** This repository contains the code for training a spiking neural network (SNN) on event-based data through a spike-timing-dependent plasticity (STDP) learning rule.
Although the focus of the project was on the visual modality (with the N-MNIST dataset), an auditory and tactile version of the code is available as well (with the datasets N-TIDIGITS and ST-MNIST).

![Network_architecture](https://user-images.githubusercontent.com/25617825/154169419-25537181-480d-46ff-9e7c-163ffd8748be.png)
*Adapted from Diehl, P. U., & Cook, M. (2015).*

This project was carried out as the master thesis on [_"Local Unsupervised Learning with Multimodal Event-Based Sensors and Spiking Neural Networks"_](http://fse.studenttheses.ub.rug.nl/id/eprint/26461) authored by Julian Lopez Gordillo and supervised by: Dr. Lyes Khacef, Prof. Dr. Elisabetta Chicca, Prof. Dr. Alejandro Linares-Barranco, Dr. Antonio Rios-Navarro, and Prof. Dr. Niels Taatgen (MSc in Artificial Intelligence 2019-2021). This project builds upon some previous work that is listed in the section _References_ later on.

# Instructions
The project was done in Python 3.8, PyTorch 1.7.1 and BindsNET 0.2.9, although newer versions will probably work just fine. 
For the complete list of dependencies check the file `requirements.txt`. 
Bindsnet is directly included as a directory and does not need to be installed.
For GPU acceleration, the appropriate graphic drivers need to be installed according to the version of PyTorch.

## Installation
1. Make sure you have a working installation of Python 3.8+ (as well as the pip package manager).
2. Clone this repository.
3. Execute `pip install -r requirements.txt` on the root folder of the repo.
4. (The downloading, unzipping, and pickling of the dataset should be handled automatically the first time you run the code. If there is any issue with the download, you can [download it](https://www.garrickorchard.com/datasets/n-mnist#h.p_ID_38) and unzip it manually (the path to the unzipped Train folder should be `.../stdp_nmnist/data/NMNIST/raw/Train`, and the equivalent for Test).
5. Everything should be ready, you can execute `python MODALITY_network.py --help` for a description of the available options. "MODALITY" should be replaced with "Visual", "Auditory", or "Tactile".  
6. For example, you can train a visual network by issuing a `python Visual_network.py [--options]` command with the specified options (adding no options will execute the default parameters).

_Note: The auditory and tactile modalities are not as polished and some things might not work as intended._

### Example
The following options are available:
```
  -h, --help            show this help message and exit
  --thresh THRESH       Threshold for the membrane voltage.
  --tc_decay TC_DECAY   Time constant for the membrane voltage decay.
  --x_tar X_TAR         Target value for the pre-synaptic trace (STDP).
  --n_neurons N_NEURONS
                        Number of neurons in the excitatory layer.
  --exc EXC             Strength of excitatory synapses.
  --inh INH             Strength of inhibitory synapses.
  --theta_plus THETA_PLUS
                        Step increase for the adaptive threshold.
  --som                 Enable for topological self-organisation.
  --n_test N_TEST       Number of samples for the testing set (if None, all are used)
  --n_train N_TRAIN     Number of samples for the training set (if None, all are used)
  --pattern_time PATTERN_TIME
                        Duration (in milliseconds) of a single pattern.
  --filename FILENAME
                        Name for the experiment (and resulting files).
  --dt DT               Simulation timestep.
  --n_epochs N_EPOCHS   Number of training epochs.
  --n_workers N_WORKERS
                        Number of parallel processes to be created.
  --seed SEED           Seed for the pseudorandom number generation.
  --progress_interval PROGRESS_INTERVAL
                        Frequency of training progress reports.
  --plot                Enable plotting (considerably slows the simulation).
  --gpu                 Enable GPU acceleration.
```

_Note: Modifying the `--patern_time` option requires having serialised the dataset files with the new pattern_time beforehand.
For example, changing the `--patern_time` to a different value than default when running `Visual_network.py` requires regenerating the dataset files with that new pattern time (to change in `NMNIST.py`).

# Future work
Throughout the code, you will find some `# PARAMETER`, `# TODO` and `# FUTURE WORK` comments:
- `# PARAMETER`, as the name suggests, tags the lines of code where important parameters are selected.
- `# TODO` marks certain improvements that I couldn't finish but would be nice additions to the code.
- `# FUTURE WORK` shows the lines of research we would have taken had we had more time for the project.

In the bigger picture, there are three levels where next steps could be taken:
1. Synapse level: this entails the exploration of new plasticity rules, something we did not contemplate in this thesis.
2. Neuron level: just like homeostasis from the ALIF model, more complex neuron models might provide other useful features that change the learning dynamics.
3. Network level: this includes adding recurrence to the network topology, increasing the number of layers or the complexity of the connectivity.

Particularly, the network level is the most simplistic one at the moment, and thus the one that would benefit the most from an update.
Apart from some scattered tasks throughout the code that could be picked up, another interesting line of research would be looking into different datasets.
For example, [LIPSFUS](https://github.com/RTC-research-group/LIPSFUS-Event-driven-dataset) is an interesting event-based audiovisual dataset where both modalities were recorded simultaneously, and therefore, are naturally synchronised. It would fit very well for this project.

# References
The module `eventvision` is slightly modified from [gorchard/event-Python](https://github.com/gorchard/event-Python) and only necessary for the reading of the data prior to serialisation.

Orchard, G., Jayawant, A., Cohen, G. K., & Thakor, N. (2015). Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades. Frontiers in Neuroscience, 9. https://doi.org/10.3389/fnins.2015.00437

Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience, 9. https://doi.org/10.3389/fncom.2015.00099

Hazan, H., Saunders, D., Sanghavi, D. T., Siegelmann, H., & Kozma, R. (2018). Unsupervised Learning with Self-Organizing Spiking Neural Networks. 2018 International Joint Conference on Neural Networks (IJCNN), 1–6. https://doi.org/10.1109/IJCNN.2018.8489673

Iyer, L. R., & Basu, A. (2017). Unsupervised learning of event-based image recordings using spike-timing-dependent plasticity. 2017 International Joint Conference on Neural Networks (IJCNN), 1840–1846. https://doi.org/10.1109/IJCNN.2017.7966074

Iyer, L. R., Chua, Y., & Li, H. (2021). Is Neuromorphic MNIST Neuromorphic? Analyzing the Discriminative Power of Neuromorphic Datasets in the Time Domain. Frontiers in Neuroscience, 15. https://doi.org/10.3389/fnins.2021.608567

