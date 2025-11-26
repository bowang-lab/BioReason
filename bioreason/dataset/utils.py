from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset
from typing import Dict, Any


def truncate_dna(
    example: Dict[str, Any], truncate_dna_per_side: int = 1024
) -> Dict[str, Any]:
    """
    Truncate DNA sequences by removing a specified number of base pairs from both ends.
    If the sequence is too short, it will return the middle portion.
    """
    for key in ["reference_sequence", "variant_sequence"]:
        sequence = example[key]
        seq_len = len(sequence)

        if seq_len > 2 * truncate_dna_per_side + 8:
            example[key] = sequence[truncate_dna_per_side:-truncate_dna_per_side]

    return example


def torch_to_hf_dataset(torch_dataset: TorchDataset) -> HFDataset:
    """
    Convert a PyTorch Dataset to a Hugging Face Dataset.

    This function takes a PyTorch Dataset and converts it to a Hugging Face Dataset
    by extracting all items and organizing them into a dictionary structure that
    can be used to create a Hugging Face Dataset.

    Args:
        torch_dataset: A PyTorch Dataset object to be converted

    Returns:
        A Hugging Face Dataset containing the same data as the input PyTorch Dataset
    """
    # Get first item to determine structure
    if len(torch_dataset) == 0:
        return HFDataset.from_dict({})

    first_item = torch_dataset[0]

    # Initialize dictionary based on first item's keys
    data_dict = (
        {k: [] for k in first_item.keys()}
        if isinstance(first_item, dict)
        else {"data": []}
    )

    # Populate dictionary
    for i in range(len(torch_dataset)):
        item = torch_dataset[i]
        if isinstance(item, dict):
            for k in data_dict:
                data_dict[k].append(item[k])
        else:
            data_dict["data"].append(item)

    return HFDataset.from_dict(data_dict)
