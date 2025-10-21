from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import torch

class DNABaseModule(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def get_dnallm_key(self):
        pass

    @abstractmethod
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        pass

    def is_embeds_input(self):
        return False
    
    @abstractmethod
    def get_processing_class(self):
        pass

    @abstractmethod
    def get_dnallm_modules_keywords(self):
        pass

    @abstractmethod
    def get_custom_multimodal_keywords(self):
        pass

    @abstractmethod
    def get_non_generate_params(self):
        pass

    @abstractmethod
    def get_custom_processing_keywords(self):
        pass

    @abstractmethod
    def prepare_prompt(self, processing_class, inputs: Dict[str, Union[torch.Tensor, Any]]):
        pass
    
    @abstractmethod
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors, padding, padding_side, add_special_tokens):
        pass