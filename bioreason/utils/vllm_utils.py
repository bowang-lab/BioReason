from typing import Optional
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import torch

from torch import nn

from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available

if is_vllm_available():
    from vllm import LLM, SamplingParams

def should_update_and_canonicalize(name: str) -> Optional[str]:
    """
    Mirror the server-side name logic:
      - only update if name starts with one of {text_model., base_model., model.}
      - strip the 'text_model.' prefix if present
    Return the canonicalized name or None to skip.
    """
    if not (name.startswith("text_model.") or name.startswith("base_model.") or name.startswith("model.")):
        return None
    if name.startswith("text_model."):
        name = name[len("text_model."):]
    return name

def fix_param_name_to_vllm(name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

def sync_fsdp1_params_to_vllm(accelerator, vllm_mode: str, vllm_client: VLLMClient, llm, module: nn.Module, prefix: str = "", visited=None):
    """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
    # For FSDP1, we need to recurse into children and also use summon_full_params
    if visited is None:
        visited = set()
    for child_name, child_module in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        sync_fsdp1_params_to_vllm(
            child_module, prefix=child_prefix, visited=visited
        )  # recurse into the child

    if isinstance(module, FSDP):
        with FSDP.summon_full_params(module, recurse=False, writeback=False):
            for param_name, param in module.named_parameters():
                full_name = f"{prefix}.{param_name}" if prefix else param_name
                full_name = fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                if full_name in visited:
                    continue  # skip FSDP subtrees already traversed
                visited.add(full_name)

                if vllm_mode == "server" and accelerator.is_main_process:
                    print("full_name", full_name)
                    vllm_client.update_named_param(full_name, param.data)
                elif vllm_mode == "colocate":
                    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights([(full_name, param.data)])

def sync_fsdp2_params_to_vllm(llm: LLM, accelerator, vllm_mode: str, vllm_client: VLLMClient, module: nn.Module):
    # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
    for name, param in module.state_dict().items():
        if param.is_cpu:
            param = param.to(torch.device("cuda"))
        param = param.full_tensor()

        if vllm_mode == "server" and accelerator.is_main_process:

            vllm_client.update_named_param(name, param)
        elif vllm_mode == "colocate":
            llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights([(name, param)])