import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from time import time as t
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import (
    all_activity,
    proportion_weighting,
    assign_labels,
)
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    # plot_assignments,
    plot_performance,
    plot_voltages,
)
from bindsnet.analysis.visualization import summary, plot_weights_movie
from modified_bindsnet import SpikingNetwork, plot_confusion_matrix, plot_weights, plot_spikes_rate, \
    plot_input_spikes, plot_assignments
from STMNIST import STMNIST
from NMNIST import NMNIST, SparseToDense

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a SNN on the STMNIST dataset.')
# Neuron parameters
parser.add_argument("--thresh", type=float, default=-52.0, help='Threshold for the membrane voltage.')
parser.add_argument("--tc_decay", type=float, default=215.0, help='Time constant for the membrane voltage decay.')
# Learning rule parameters
parser.add_argument("--x_tar", type=float, default=0.4, help='Target value for the pre-synaptic trace (STDP).')
# Network parameters
parser.add_argument("--n_neurons", type=int, default=100, help='Number of neurons in the excitatory layer.')
parser.add_argument("--exc", type=float, default=22.5, help='Strength of excitatory synapses.')
parser.add_argument("--inh", type=float, default=17.5, help='Strength of inhibitory synapses.')
parser.add_argument("--theta_plus", type=float, default=0.2, help='Step increase for the adaptive threshold.')
parser.add_argument("--som", dest="som", action="store_true", help='Enable for topological self-organisation.')
# Data parameters
parser.add_argument("--n_test", type=int, default=None, help='Number of samples for the testing set (if None, '
                                                             'all are used)')
parser.add_argument("--n_train", type=int, default=None, help='Number of samples for the training set (if None, '
                                                              'all are used)')
parser.add_argument("--pattern_time", type=int, default=2000, help='Duration (in milliseconds) of a single pattern.')
parser.add_argument("--filename", type=str, default='test', help='Name for the experiment (and resulting files).')
# Simulation parameters
parser.add_argument("--dt", type=float, default=1.0, help='Simulation timestep.')
parser.add_argument("--n_epochs", type=int, default=1, help='Number of training epochs.')
parser.add_argument("--n_workers", type=int, default=-1, help='Number of parallel processes to be created.')
parser.add_argument("--seed", type=int, default=0, help='Seed for the pseudorandom number generation.')
parser.add_argument("--progress_interval", type=int, default=10, help='Frequency of training progress reports.')
parser.add_argument("--plot", dest="plot", action="store_true", help='Enable plotting (considerably slows the '
                                                                     'simulation).')
parser.add_argument("--gpu", dest="gpu", action="store_true", help='Enable GPU acceleration.')
# Defaults
parser.set_defaults(plot=False, gpu=False, som=False)

args = parser.parse_args()

thresh = args.thresh
tc_decay = args.tc_decay
x_tar = args.x_tar
n_neurons = args.n_neurons
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
som = args.som
n_test = args.n_test
n_train = args.n_train
pattern_time = args.pattern_time
filename = args.filename
dt = args.dt
n_epochs = args.n_epochs
n_workers = args.n_workers
seed = args.seed
progress_interval = args.progress_interval
plot = args.plot
gpu = args.gpu

# Create directories
directories = ["results", "results/" + filename]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Set up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
np.random.seed(seed)

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

# Load training data
train_dataset = STMNIST(root=os.path.join("./", "data"), download=True, train=True, dt=dt, transform=SparseToDense())

# Declare auxiliary variables and parameters
n_classes = 10
n_train = len(train_dataset) if n_train == None else n_train
update_interval = n_train // 60
data_dim_sq = train_dataset[0][0].shape[1]
data_dim = int(np.sqrt(data_dim_sq))
data_dim_sqrt = int(np.sqrt(data_dim))
n_neurons_sqrt = int(np.ceil(np.sqrt(n_neurons)))
c_inhib = torch.linspace(-5.0, -17.5, n_train // update_interval, device=device)
w_inhib = (torch.ones(n_neurons, n_neurons) - torch.diag(torch.ones(n_neurons))).to(device)
weights_mask = (1 - torch.diag(torch.ones(n_neurons))).to(device)
pattern_repetition_counter = 0

# Build the network
network = SpikingNetwork(n_neurons=n_neurons, inpt_shape=(1, data_dim, data_dim), n_inpt=data_dim_sq, dt=dt,
                         thresh=thresh, tc_decay=tc_decay, theta_plus=theta_plus, x_tar=x_tar,
                         weight_factor=1.0, exc=exc, inh=inh, som=som, start_inhib=-5.0, max_inhib=-17.5)
if gpu:
    network.to("cuda")
print(summary(network))

# Record spikes during the simulation
excitatory_spikes = torch.tensor((int(pattern_time / dt), n_neurons), dtype=torch.bool, device=device)
spike_record = torch.zeros((update_interval, int(pattern_time / dt), n_neurons), device=device)
cumulative_spikes = torch.zeros((n_train // update_interval, n_neurons), device=device)

# Neuron assignments and spike proportions
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates
accuracy = {"all": [], "proportion": []}
proportion_pred = torch.zeros(update_interval, dtype=torch.int64, device=device)
label_tensor = torch.zeros(update_interval, dtype=torch.int64, device=device)
training_proportion_pred = torch.zeros(0, dtype=torch.int64, device=device)
training_label_tensor = torch.zeros(0, dtype=torch.int64, device=device)

# Set up monitors for spikes and voltages
exc_voltage_monitor = Monitor(
    network.layers["Excitatory"], ["v"], time=int(pattern_time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Inhibitory"], ["v"], time=int(pattern_time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(pattern_time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"Input"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(pattern_time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
cm_ax = None
cspikes_ax = None
input_s_im = None

# Train the network
print("\nBegin training.\n")
start = t()
for epoch in range(n_epochs):
    labels = []
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Growing inhibition strategy
        # elif network.som and (step > 0 and step % update_interval == 0):
        #     for i in range(network.n_neurons):
        #         for j in range(network.n_neurons):
        #             if i != j:
        #                 x1, y1 = i // network.n_sqrt, i % network.n_sqrt
        #                 x2, y2 = j // network.n_sqrt, j % network.n_sqrt
        #
        #                 w_inhib[i, j] = max(network.max_inhib, c_inhib[(step - 1) // update_interval] *
        #                                     network.inhib_scaling * np.sqrt(euclidean([x1, y1], [x2, y2])))
        #     network.connections['Inhibitory', 'Excitatory'].w.copy_(w_inhib)
        # Two-level inhibition strategy
        elif network.som and step == 0.1 * n_train:
            w = -network.inh * (torch.ones(network.n_neurons, network.n_neurons)
                                - torch.diag(torch.ones(network.n_neurons)))
            network.connections['Inhibitory', 'Excitatory'].w.copy_(w_inhib)

        # Get next input sample
        inputs = {"Input": batch[0].view(int(pattern_time / dt), 1, 1, data_dim, data_dim)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Progress updates
        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions
            all_activity_pred = all_activity(
                spikes=spike_record,
                assignments=assignments,
                n_labels=n_classes,
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )
            tqdm.write(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            tqdm.write(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )
            # Assign labels to excitatory layer neurons
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )
            # FUTURE WORK: Limit the amount of labels used for the neuron assignments, and compare how the
            # performance of the network varies against the % of labels used.

            training_proportion_pred = torch.cat((training_proportion_pred, proportion_pred), dim=0)
            training_label_tensor = torch.cat((training_label_tensor, label_tensor), dim=0)
            labels = []

        labels.append(batch[1])

        # Run the network on the input
        network.connections['Input', 'Excitatory'].weight_factor = 1.0
        for spikes_check in range(5):
            network.run(inputs=inputs, time=pattern_time, input_time_dim=1)

            # Get voltage recording
            exc_voltages = exc_voltage_monitor.get("v")
            inh_voltages = inh_voltage_monitor.get("v")

            # If not enough spikes, present that sample again (with an increased weight factor)
            excitatory_spikes = spikes["Excitatory"].get("s").squeeze()
            if excitatory_spikes.sum().sum() < 2:
                network.connections['Input', 'Excitatory'].weight_factor *= 1.2
                pattern_repetition_counter += 1
            else:
                break

        # Add to spikes recording
        spike_record[step % update_interval].copy_(excitatory_spikes, non_blocking=True)

        # Optionally plot simulation information
        if plot:
            input_exc_weights = network.connections[("Input", "Excitatory")].w
            square_weights = get_square_weights(input_exc_weights.view(data_dim_sq, n_neurons), n_neurons_sqrt,
                                                data_dim)
            square_assignments = get_square_assignments(assignments, n_neurons_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Excitatory": exc_voltages, "Inhibitory": inh_voltages}
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line")
            input_spikes = torch.sum(spikes['Input'].get("s").squeeze(), 0)
            input_s_im = plot_input_spikes(input_spikes, im=input_s_im)

            plt.pause(1e-8)

        # Save plots at checkpoints and at the end of the training
        if (step >= (n_train - 1)) or (step % update_interval == 0 and step > 0):
            cumulative_spikes[min(59, (step - 1) // update_interval), :].add_(torch.sum(spike_record.long(), (0, 1)))
            input_exc_weights = network.connections[("Input", "Excitatory")].w
            square_weights = get_square_weights(input_exc_weights.view(data_dim_sq, n_neurons), n_neurons_sqrt,
                                                data_dim)
            square_assignments = get_square_assignments(assignments, n_neurons_sqrt)
            plot_weights(square_weights, im=weights_im,
                         save=f'./results/{filename}/weights_{filename}.png')
            plot_assignments(square_assignments, im=assigns_im, save=f'./results/{filename}/assignments'
                                                                     f'_{filename}.png')
            plot_performance(accuracy, x_scale=update_interval, ax=perf_ax,
                             save=f'./results/{filename}/performance_{filename}.png')
            plot_confusion_matrix(torch.Tensor.cpu(training_proportion_pred), torch.Tensor.cpu(training_label_tensor),
                                  save=f'./results/{filename}/confusion_matrix_{filename}.png')
            plot_spikes_rate(cumulative_spikes, save=f'./results/{filename}/cumulative_spikes_{filename}.png',
                             update_interval=update_interval)
            torch.save(network, f'./results/{filename}/model_{filename}.pt')
            network.save(f'./results/{filename}/network_{filename}.npz')

        network.reset_state_variables()  # Reset state variables

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Load testing data
test_dataset = STMNIST(root=os.path.join("./", "data"), download=True, train=False, dt=dt,
                       transform=SparseToDense())
n_test = len(test_dataset) if n_test == None else n_test

# Sequence of accuracy estimates
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation
spike_record = torch.zeros((1, int(pattern_time / dt), n_neurons), device=device)
testing_proportion_pred = torch.zeros(n_test, dtype=torch.int64, device=device)
testing_label_tensor = torch.zeros(n_test, dtype=torch.int64, device=device)

# Test the network
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break

    # Get next input sample
    inputs = {"Input": batch[0].view(int(pattern_time / dt), 1, 1, data_dim, data_dim)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input
    network.connections['Input', 'Excitatory'].weight_factor = 1.2
    for spikes_check in range(5):

        network.run(inputs=inputs, time=pattern_time, input_time_dim=1)

        excitatory_spikes = spikes["Excitatory"].get("s").squeeze()
        # If not enough spikes, present that sample again (with increased weight factor)
        if excitatory_spikes.sum().sum() < 2:
            network.connections['Input', 'Excitatory'].weight_factor *= 1.2
        else:
            break

    # Add to spikes recording
    spike_record[0].copy_(excitatory_spikes, non_blocking=True)

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch[1], device=device)

    # Get network predictions
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    testing_proportion_pred[step] = proportion_pred
    testing_label_tensor[step] = label_tensor

    network.reset_state_variables()  # Reset state variables
    pbar.set_description_str("Test progress: ")
    pbar.update()

plot_confusion_matrix(torch.Tensor.cpu(testing_proportion_pred), torch.Tensor.cpu(testing_label_tensor),
                      save=f'./results/{filename}/test_confusion_matrix_{filename}.png')

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

# Save logs.
with open(f'./results/{filename}/log_{filename}.txt', 'w+') as log_file:
    print(f'''
    All activity accuracy: {accuracy["all"] / n_test}
    Proportion weighting accuracy: {accuracy["proportion"] / n_test} \n
    Progress: {epoch + 1} / {n_epochs} ({t() - start} seconds)
    Testing complete.\n
    Pattern repetitions during training: {pattern_repetition_counter}\n''', file=log_file)
