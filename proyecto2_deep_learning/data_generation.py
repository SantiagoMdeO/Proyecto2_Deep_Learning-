import torch
import pandas as pd
from sdv.single_table import CTGANSynthesizer

# Patch torch.load to remap MPS tensors to CPU for environments without Apple Silicon
_original_torch_load = torch.load

def _cpu_map_load(*args, **kwargs):
    kwargs.setdefault('map_location', 'cpu')
    return _original_torch_load(*args, **kwargs)

torch.load = _cpu_map_load


def generate_synthetic_training_data(n=30_000):
    """Generates synthetic training data using pre-trained CTGAN models for each credit score category.

    Args:
        n (int, optional): The number of samples to generate for each category. Defaults to 30_000.
    Returns:
        pd.DataFrame: The generated synthetic training data.
    """
    good_generator = CTGANSynthesizer.load("../models/v4/synth_good.pkl")
    poor_generator = CTGANSynthesizer.load("../models/v4/synth_poor.pkl")
    standard_generator = CTGANSynthesizer.load("../models/v4/synth_standard.pkl")

    synth_good = good_generator.sample(n)
    synth_poor = poor_generator.sample(n)
    synth_standard = standard_generator.sample(n)

    full_data = pd.concat([synth_good, synth_poor, synth_standard], ignore_index=True)
    shuffled_data = full_data.sample(frac=1).reset_index(drop=True)
    return shuffled_data