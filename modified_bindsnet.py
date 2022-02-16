from typing import Optional, Union, Tuple, List, Sequence, Iterable, Type, Dict, Sized
import numpy as np
import torch
from bindsnet.learning.reward import AbstractReward
from bindsnet.models import DiehlAndCook2015
from bindsnet.utils import im2col_indices
from matplotlib import pyplot as plt, ticker
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import euclidean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, multilabel_confusion_matrix
from torch.nn import Parameter
from torch.nn.modules.utils import _pair
import torch.nn as nn
from torchvision import models
from bindsnet.learning import PostPre, WeightDependentPostPre, LearningRule
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes, AdaptiveLIFNodes, Nodes
from bindsnet.network.topology import Connection, LocalConnection, AbstractConnection, Conv2dConnection


class FactoredConnection(Connection):
    def __init__(
            self,
            source: Nodes,
            target: Nodes,
            nu: Optional[Union[float, Sequence[float]]] = None,
            reduction: Optional[callable] = None,
            weight_decay: float = 0.0,
            weight_factor: float = 1.0,
            **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object, slightly modified to include a factor multiplying the weights to
        promote post-synaptic spikes.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        :param weight_factor: Factor for the weights.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)
        self.weight_factor = weight_factor

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights and weight factor.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = s.view(s.size(0), -1).float() @ (self.w * self.weight_factor)
        else:
            post = s.view(s.size(0), -1).float() @ (self.w * self.weight_factor) + self.b
        return post.view(s.size(0), *self.target.shape)


class DiehlCookSTDP(LearningRule):
    # language=rst
    """
    STDP rule from Diehl&Cook2015, involving only post-synaptic spiking activity. The post-synaptic
    update is positive and is dependent on the magnitude of the synaptic weights and on the pre-synaptic trace.
    """

    def __init__(
            self,
            connection: AbstractConnection,
            nu: Optional[float] = None,
            reduction: Optional[callable] = None,
            weight_decay: float = 0.0,
            **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for ``WeightDependentPostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``WeightDependentPostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        :param x_tar: Constant target for the pre-synaptic trace. Controls the potentiation-depression balance.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs
        )

        assert self.source.traces, "Pre-synaptic nodes must record spike traces."
        assert (
                connection.wmin != -np.inf and connection.wmax != np.inf
        ), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax
        self.x_tar = 0.4

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Diehl&Cook's learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()

        update = 0

        # Post-synaptic update.
        if self.nu is not None:
            outer_product = self.reduction(torch.where(target_s.bool(), source_x - self.x_tar,
                                                       torch.zeros_like(source_x)), dim=0)
            update += self.nu[1] * outer_product * torch.pow((self.wmax - self.connection.w), 0.2)

        self.connection.w += update

        super().update()


class IyerAndBasu2017(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_. Some modifications are being implemented
    on the lines of the work from `(Iyer & Basu 2017) <https://ieeexplore.ieee.org/document/7966074>`_, and also from
    `(Hazan et al 2018) <https://arxiv.org/abs/1807.09374>`_..
    """

    def __init__(
            self,
            n_inpt: int,
            n_neurons: int = 100,
            exc: float = 22.5,
            inh: float = 17.5,
            dt: float = 1.0,
            nu: Optional[float] = 5e-2,
            x_tar: float = 0.4,
            reduction: Optional[callable] = None,
            wmin: float = 0.0,
            wmax: float = 1.0,
            winit: float = 0.3,
            norm: float = None,
            thresh: float = -52.0,
            tc_decay: float = 20.0,
            theta_plus: float = 0.2,
            tc_theta_decay: float = 1e7,
            inpt_shape: Optional[Iterable[int]] = None,
            weight_factor: float = 1.0,
            som: bool = False,
            start_inhib: float = -0.1,
            max_inhib: float = -20.0,
            inhib_scaling: float = 0.75,
    ) -> None:
        # language=rst
        """
        Constructor for class ``IyerAndBasu2017``.
        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param winit: Mean value for weight initialisation input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param thresh: Voltage for ``DiehlAndCookNodes`` firing threshold.
        :param tc_decay: Time constant of ``DiehlAndCookNodes`` membrane voltage decay.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        :param weight_factor: The factor for the input-excitatory connection weights, to promote spiking activity.
        :param som: Option to enable the self-organising property (triggers a different inhibitory-excitatory weight
            distribution.
        :param start_inhib: Minimum weights to be set between the inhibitory and excitatory layer (with SOM).
        :param max_inhib: Maximum weights allowed between the inhibitory and excitatory layer (with SOM).
        :param inhib_scaling: Factor that controls the radius of the inhibition increase.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
        self.n_sqrt = np.sqrt(n_neurons)
        self.som = som
        self.start_inhib = start_inhib
        self.max_inhib = max_inhib
        self.inhib_scaling = inhib_scaling

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True,
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,  # PARAMETER
            traces=False,
            traces_additive=False,
            rest=-65.0,
            reset=-60.0,
            thresh=thresh,  # PARAMETER
            refrac=5,
            tc_decay=tc_decay,  # PARAMETER
            tc_trace=20.0,
            theta_plus=theta_plus,  # PARAMETER
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(n=self.n_neurons, traces=False, rest=-60.0, reset=-45.0, thresh=-40.0, tc_decay=10.0,
                             refrac=2, tc_trace=20.0)

        # Connections
        w = winit * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = FactoredConnection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=DiehlCookSTDP,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            weight_factor=weight_factor,  # PARAMETER
        )
        input_exc_conn.update_rule.x_tar = x_tar  # PARAMETER
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc)

        if self.som:
            w = torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons))
            for i in range(self.n_neurons):
                for j in range(self.n_neurons):
                    if i != j:
                        x1, y1 = i // self.n_sqrt, i % self.n_sqrt
                        x2, y2 = j // self.n_sqrt, j % self.n_sqrt

                        w[i, j] = max(self.max_inhib, self.start_inhib * self.inhib_scaling * np.sqrt(euclidean(
                            [x1, y1], [x2, y2])))
            inh_exc_conn = Connection(source=inh_layer, target=exc_layer, w=w)
        else:
            w = -self.inh * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons)))
            inh_exc_conn = Connection(source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0)

        # Add to network
        self.add_layer(input_layer, name="Input")
        self.add_layer(exc_layer, name="Excitatory")
        self.add_layer(inh_layer, name="Inhibitory")
        self.add_connection(input_exc_conn, source="Input", target="Excitatory")
        self.add_connection(exc_inh_conn, source="Excitatory", target="Inhibitory")
        self.add_connection(inh_exc_conn, source="Inhibitory", target="Excitatory")


# Plotting functions
def plot_confusion_matrix(
        predicted: torch.LongTensor,
        correct: torch.LongTensor,
        save: Optional[str] = None,
) -> Axes:
    # language=rst
    """
    Plot confusion matrix for the network's performance.

    :param predicted: Tensor with the predicted labels.
    :param correct: Tensor with the ground truth labels (must have the same dimensions than predicted).
    :param save: file name to save fig, if None = not saving fig.
    :return: Used for re-drawing the performance plot.
    """

    if save is not None:
        plt.ioff()

        cm = confusion_matrix(correct, predicted, normalize=None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(include_values=True, cmap='viridis', xticks_rotation='horizontal', values_format=None,
                  colorbar=True)

        plt.savefig(save, bbox_inches="tight")
        plt.close()
        plt.ion()
    else:
        cm = confusion_matrix(correct, predicted, normalize=None)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(include_values=True, cmap='viridis', xticks_rotation='horizontal', values_format=None,
                  colorbar=True)

    return disp


def plot_weights(
        weights: torch.Tensor,
        wmin: Optional[float] = 0,
        wmax: Optional[float] = 1,
        im: Optional[AxesImage] = None,
        figsize: Tuple[int, int] = (5, 5),
        cmap: str = "hot_r",
        save: Optional[str] = None,
) -> AxesImage:
    # language=rst
    """
    Plot a connection weight matrix.

    :param weights: Weight matrix of ``Connection`` object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    :param im: Used for re-drawing the weights plot.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :param save: file name to save fig, if None = not saving fig.
    :return: ``AxesImage`` for re-drawing the weights plot.
    """
    local_weights = weights.detach().clone().cpu().numpy()
    if save is not None:
        plt.ioff()

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(local_weights, cmap=cmap, vmin=wmin, vmax=wmax)
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_aspect("auto")

        plt.colorbar(im, cax=cax)
        fig.tight_layout()

        plt.savefig(save, bbox_inches="tight")

        plt.close(fig)
        plt.ion()
        return im
    else:
        if not im:
            fig, ax = plt.subplots(figsize=figsize)

            im = ax.imshow(local_weights, cmap=cmap, vmin=wmin, vmax=wmax)
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)

            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_aspect("auto")

            plt.colorbar(im, cax=cax)
            fig.tight_layout()
        else:
            im.set_data(local_weights)

        return im


def plot_cumulative_spikes(
        spikes: torch.Tensor,
        im: Optional[AxesImage] = None,
        figsize: Tuple[int, int] = (5, 5),
        cmap: str = "YlGn",
        save: Optional[str] = None,
        update_interval: int = 600,
) -> AxesImage:
    # language=rst
    """
    Plot the spikes generated in the excitatory layer (binned in update intervals).

    :param weights: Weight matrix of ``Connection`` object.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :param save: file name to save fig, if None = not saving fig.
    :return: ``AxesImage`` for re-drawing the cumulative spikes plot.
    """
    local_spikes = spikes.detach().clone().cpu().numpy()
    (x_dim, y_dim) = local_spikes.shape
    if save is not None:
        plt.ioff()

        fig, ax = plt.subplots()
        im = plt.matshow(local_spikes)

        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * update_interval))
        im.axes.yaxis.set_major_formatter(ticks_y)

        plt.title("Spikes elicited in the excitatory layer")
        plt.xlabel("Neurons")
        plt.ylabel("Training patterns")

        plt.colorbar(im)
        fig.tight_layout()

        plt.savefig(save, bbox_inches="tight")

        plt.close(fig)
        plt.ion()
        return im
    else:
        if not im:
            fig, ax = plt.subplots()
            im = plt.matshow(local_spikes)

            ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * update_interval))
            im.axes.yaxis.set_major_formatter(ticks_y)

            plt.title("Spikes elicited in the excitatory layer")
            plt.xlabel("Neurons")
            plt.ylabel("Training patterns")

            plt.colorbar(im)
            fig.tight_layout()

        else:
            im.set_data(local_spikes)

        return im


def plot_input_spikes(
        spikes: torch.Tensor,
        im: Optional[AxesImage] = None,
        figsize: Tuple[int, int] = (5, 5),
) -> AxesImage:
    # language=rst
    """
    Plot the spikes generated in the excitatory layer (binned in update intervals).

    :param weights: Weight matrix of ``Connection`` object.
    :param figsize: Horizontal, vertical figure size in inches.
    :param cmap: Matplotlib colormap.
    :return: ``AxesImage`` for re-drawing the cumulative spikes plot.
    """
    local_spikes = spikes.detach().clone().cpu().numpy()
    max_spike_count = local_spikes.max()

    if not im:
        fig, ax = plt.subplots()
        im = plt.matshow(local_spikes)

        plt.title(f"Maximum spike count is: {max_spike_count}")

        plt.colorbar(im)
        fig.tight_layout()

    else:
        plt.title(f"Maximum spike count is: {max_spike_count}")
        im.set_data(local_spikes)

    return im