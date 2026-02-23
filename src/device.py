import torch

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def device_info():
    return f"torch={torch.__version__} mps_available={torch.backends.mps.is_available()} mps_built={torch.backends.mps.is_built()}"
