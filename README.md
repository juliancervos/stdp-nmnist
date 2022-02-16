# Summary
**In short:** This repository contains the code for training a spiking neural network (SNN) on event-based data (and specifically, on the dataset N-MNIST) through spike-timing-dependent plasticity (STDP) learning rule.

![Network_architecture](https://user-images.githubusercontent.com/25617825/154169419-25537181-480d-46ff-9e7c-163ffd8748be.png)


This project was carried out as the master thesis on _"Local Unsupervised Learning with Multimodal Event-Based Sensors and Spiking Neural Networks"_ authored by Julian Lopez Gordillo and supervised by: Dr. Lyes Khacef, Prof. Dr. Elisabetta Chicca, Prof. Dr. Alejandro Linares-Barranco, Dr. Antonio Rios-Navarro, and Prof. Dr. Niels Taatgen (MSc in Artificial Intelligence 2019-2021). This project builds upon some previous work that is listed in the section _References_ later on.

# Instructions
The project was done in Python 3.8, PyTorch 1.7.1 and BindsNET 0.2.7, although newer versions will probably work just fine. For the complete list of dependencies check the file `requirements.txt`.

## Installation
1. Make sure you have a working installation of Python 3.8+ (as well as the pip package manager).
2. Clone this repository.
3. Execute `pip -r install requirements.txt` on the root folder of the repo.
4. (The downloading, unzipping, and pickling of the dataset should be handled automatically the first time you run the code. If there is any issue with the download, you can [download it](https://www.garrickorchard.com/datasets/n-mnist#h.p_ID_38) and unzip it manually (the path to the unzipped Train folder should be `.../stdp_nmnist/data/NMNIST/raw/Train`, and the equivalent for Test).
5. Everything should be ready, you can execute `python Visual_network.py --help` for a description of the available options.  
6. You can train a network by issuing a `python Visual_network.py [--options]` command with the specified options (adding no options will execute the default parameters).

# References
The module `eventvision` is slightly modified from [gorchard/event-Python](https://github.com/gorchard/event-Python) and only necessary for the reading of the data prior to serialisation.

Orchard, G., Jayawant, A., Cohen, G. K., & Thakor, N. (2015). Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades. Frontiers in Neuroscience, 9. https://doi.org/10.3389/fnins.2015.00437

Diehl, P. U., & Cook, M. (2015). Unsupervised learning of digit recognition using spike-timing-dependent plasticity. Frontiers in Computational Neuroscience, 9. https://doi.org/10.3389/fncom.2015.00099

Hazan, H., Saunders, D., Sanghavi, D. T., Siegelmann, H., & Kozma, R. (2018). Unsupervised Learning with Self-Organizing Spiking Neural Networks. 2018 International Joint Conference on Neural Networks (IJCNN), 1–6. https://doi.org/10.1109/IJCNN.2018.8489673

Iyer, L. R., & Basu, A. (2017). Unsupervised learning of event-based image recordings using spike-timing-dependent plasticity. 2017 International Joint Conference on Neural Networks (IJCNN), 1840–1846. https://doi.org/10.1109/IJCNN.2017.7966074

Iyer, L. R., Chua, Y., & Li, H. (2021). Is Neuromorphic MNIST Neuromorphic? Analyzing the Discriminative Power of Neuromorphic Datasets in the Time Domain. Frontiers in Neuroscience, 15. https://doi.org/10.3389/fnins.2021.608567

