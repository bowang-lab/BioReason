from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
from typing import Dict, Any, Union, List, Optional, Callable, Type
from trl.data_utils import maybe_apply_chat_template
import torch

import re

from bioreason.dna_modules.dna_module import DNABaseModule
from bioreason.models.dna_llm import DNALLMModel
from bioreason.models.dl.processing_dl import DLProcessor


class NucleotideDNAModule(DNABaseModule):
    """
    DNA module implementation for NucleotideTransformer-based models.

    This module provides the interface between DNA-LLM models and the training
    infrastructure, handling model loading, processing setup, and reward functions.
    """

    def __init__(self):
        """Initialize the NucleotideDNAModule."""
        super().__init__()

    def post_model_init(self, model: Any, processing_class: Any) -> None:
        """
        Perform any post-initialization setup on the model.

        Args:
            model: The initialized model
            processing_class: The processor for the model
        """
        # No post-init needed for this implementation
        pass

    def get_processing_class(self) -> Type:
        """
        Get the processing class to use with this DNA-LLM model.

        Returns:
            The processing class
        """
        return DLProcessor

    def get_dnallm_modules_keywords(self) -> List[str]:
        """
        Get keywords to identify DNA-specific modules in the model.

        Used to exclude DNA modules from LoRA adaptation during training.

        Returns:
            List of keywords that identify DNA modules
        """
        return ["dna"]

    def get_custom_multimodal_keywords(self) -> List[str]:
        """
        Get keywords for multimodal inputs that should be passed to the model.

        Returns:
            List of input keywords for multimodal processing
        """
        return ["dna_tokenized", "batch_idx_map"]

    def get_non_generate_params(self) -> List[str]:
        """
        Get parameter names that should be excluded from generation.

        Returns:
            List of parameter names to exclude from generation calls
        """
        return ['answer', 'prompt']

    def get_custom_processing_keywords(self) -> List[tuple]:
        """
        Get custom processing keywords for the processor.

        Returns:
            List of (component, parameter) tuples for custom processing
        """
        return [("dna_tokenizer", "max_length")]

    def prepare_prompt(
        self, processing_class: Any, inputs: List[Dict[str, Union[torch.Tensor, Any]]]
    ) -> List[str]:
        """
        Prepare prompts from input examples.

        Args:
            processing_class: The processor to use
            inputs: List of input examples

        Returns:
            List of prepared prompts
        """
        prompts_text = [
            maybe_apply_chat_template(example, processing_class)["prompt"]
            for example in inputs
        ]
        return prompts_text

    def prepare_model_inputs(
        self,
        processing_class: Any,
        model: Any,
        prompts_text: List[str],
        batch_dna_sequences: List[List[str]],
        return_tensors: str = "pt",
        padding: bool = True,
        padding_side: str = "left",
        add_special_tokens: bool = False,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for the model.

        Args:
            processing_class: The processor to use
            model: The model to prepare inputs for
            prompts_text: List of text prompts
            batch_dna_sequences: List of lists of DNA sequences
            return_tensors: Return format for tensors
            padding: Whether to pad inputs
            padding_side: Side to pad on
            add_special_tokens: Whether to add special tokens

        Returns:
            Processed inputs for the model
        """
        # Handle DataParallel wrapped models by accessing the module attribute if needed
        max_length_text = model.max_length_text if not hasattr(model, 'module') else model.module.max_length_text
        max_length_dna = model.max_length_dna if not hasattr(model, 'module') else model.module.max_length_dna
        
        prompt_inputs = processing_class(
            text=prompts_text,
            batch_dna_sequences=batch_dna_sequences,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
            max_length_text=max_length_text,
            max_length_dna=max_length_dna,
        )

        return prompt_inputs

    def is_embeds_input(self) -> bool:
        """
        Whether the model uses embeddings as input (instead of token IDs).

        Returns:
            Boolean indicating if the model takes embedding inputs
        """
        return True

    @staticmethod
    def get_question_template() -> str:
        """
        Get the template for formatting questions.

        Returns:
            String template for questions
        """
        return "{Question}"

    @staticmethod
    def _extract_xml_answer(text: str) -> str:
        answer = text.split("</think>")[-1]
        return answer.strip()

    # Reward functions
    @staticmethod
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [NucleotideDNAModule._extract_xml_answer(r) for r in responses]
        # extracted_responses = [r.lower().replace("answer:", "").strip() for r in extracted_responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if a.lower() in r.lower() else 0.0 for r, a in zip(extracted_responses, answer)]
    
    @staticmethod
    def concise_reward_func(completions, **kwargs) -> List[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [NucleotideDNAModule._extract_xml_answer(r) for r in responses]
        return [0.5 if len(r.split(' ')) <= 10 else 0.0 for r in extracted_responses]
    
    @staticmethod
    def strict_format_reward_func(completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"^<think>\n.*?\n</think>\n.*?\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    @staticmethod
    def soft_format_reward_func(completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r"<think>.*?</think>\s*.*?"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]
    
    @staticmethod
    def xmlcount_reward_func(completions, **kwargs) -> List[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [NucleotideDNAModule._count_xml(c) for c in contents]

    @staticmethod
    def _count_xml(text) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("\n</think>\n") == 1:
            count += 0.125
        return count

    @staticmethod
    def format_reward_rec(completions: List[Dict[str, Any]], **kwargs) -> List[float]:
        """
        Check if the Qwen model output matches a specific format.

        Args:
            completions: List of model completions
            **kwargs: Additional arguments

        Returns:
            List of reward scores (1.0 for match, 0.0 for no match)
        """
        import re
        import os
        from datetime import datetime

        # Pattern to match the expected output format
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.search(pattern, content, re.DOTALL) is not None
            for content in completion_contents
        ]

        # Log format results if in debug mode
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(
                log_path.replace(".txt", "_format.txt"), "a", encoding="utf-8"
            ) as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")

        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def select_reward_func(func: str, task_type: str) -> Callable:
        """
        Select the appropriate reward function based on function name and task type.

        Args:
            func: The type of reward function ('accuracy', 'format', etc.)
            task_type: The type of task ('rec', etc.)

        Returns:
            The reward function to use

        Raises:
            ValueError: If the function or task type is not supported
        """
        if func == "accuracy":
            match task_type:
                case "rec":
                    return NucleotideDNAModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return NucleotideDNAModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")