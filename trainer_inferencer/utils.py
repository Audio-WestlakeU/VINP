import importlib
import os
import time
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import *


class ExecutionTime:
    """Count execution time.

    Examples:
        timer = ExecutionTime()
        ...
        print(f"Finished in {timer.duration()} seconds.")
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_module(path: str, args: Optional[dict] = None, initialize: bool = True):
    """Load module or function dynamically with "args".

    Args:
        path: module path in this project.
        args: parameters that will be passed to the Class or the Function in the module.
        initialize: whether to initialize the Class or the Function with args.

    Examples:
        Config items are as follows:

            [model]
            path = "model.FullSubNetModel"
            [model.args]
            n_frames = 32
            ...

        This function will:
            1. Load the "model.full_sub_net" module.
            2. Call "FullSubNetModel" Class (or Function) in "model.full_sub_net" module.
            3. If initialize is True:
                instantiate (or call) the Class (or the Function) and pass the parameters (in "[model.args]") to it.
    """
    module_path = ".".join(path.split(".")[:-1])
    class_or_function_name = path.split(".")[-1]

    module = importlib.import_module(module_path)
    class_or_function = getattr(module, class_or_function_name)

    if initialize:
        if args:
            return class_or_function(**args)
        else:
            return class_or_function()
    else:
        return class_or_function


def print_tensor_info(tensor, flag="Tensor"):
    def floor_tensor(float_tensor):
        return int(float(float_tensor) * 1000) / 1000

    print(
        f"{flag}\n"
        f"\t"
        f"max: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, "
        f"mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}"
    )


def set_requires_grad(nets, requires_grad=False):
    """Set "requies_grad=Fasle" for all the networks to avoid unnecessary computations.

    Args:
        nets: list of networks
        requires_grad
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def prepare_device(n_gpu: int, keep_reproducibility=False):
    """Choose to use CPU or GPU depend on the value of "n_gpu".

    Args:
        n_gpu(int): the number of GPUs used in the experiment. if n_gpu == 0, use CPU; if n_gpu >= 1, use GPU.
        keep_reproducibility (bool): if we need to consider the repeatability of experiment, set keep_reproducibility to True.

    See Also
        Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        # possibly at the cost of reduced performance
        if keep_reproducibility:
            print("Using CuDNN deterministic mode in the experiment.")
            # ensures that CUDA selects the same convolution algorithm each time
            torch.backends.cudnn.benchmark = False
            # configures PyTorch only to use deterministic implementation
            torch.set_deterministic(True)
        else:
            # causes cuDNN to benchmark multiple convolution algorithms and select the fastest
            torch.backends.cudnn.benchmark = True

        device = torch.device("cuda:0")

    return device


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def basename(path):
    filename, ext = os.path.splitext(os.path.basename(path))
    return filename, ext


def plot_spectrogram(spectrogram):
    # spectrogram += 1e-8
    # spectrogram = np.log(spectrogram)
    # spectrogram[spectrogram < -8] = -8
    # spectrogram[spectrogram < -8] = -8
    # spectrogram[spectrogram > 8] = 8
    spectrogram = np.abs(spectrogram)
    spectrogram = np.clip(spectrogram,1e-4,100)
    fig = plt.figure()
    plt.imshow(np.log10(spectrogram), aspect="auto", origin="lower", vmin=-4)
    plt.colorbar()
    plt.close()

    return fig


def set_optimizer(models, optimizer_config, scheduler_config):

    optimizer_type = optimizer_config["type"]
    scheduler_type = scheduler_config["type"]
    optimizer_args = optimizer_config["args"]
    scheduler_args = scheduler_config["args"]

    # parameters = list(models.parameters())
    # filter(lambda p: p.requires_grad, models.parameters())

    optimizer_class = getattr(torch.optim, optimizer_type)
    optimizer = optimizer_class(
        filter(lambda p: p.requires_grad, models.parameters()), **optimizer_args
    )

    if scheduler_type and len(scheduler_args) > 0:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_type)
        scheduler = scheduler_class(optimizer, **scheduler_args)
        return optimizer, scheduler
    else:
        return optimizer, None


def freeze_layers_with_name_contains(model, keyword):
    for name, param in model.modules.named_parameters():
        if keyword in name:
            param.requires_grad = False

def _get_num_params(model: torch.nn.Module):
    num_params = sum(param.numel() for param in model.parameters())
    return num_params