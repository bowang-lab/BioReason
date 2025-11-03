# dna_vllm.py
import os
import glob
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM

from vllm import LLM, SamplingParams
from safetensors import safe_open

from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE
from bioreason.models.evo2_tokenizer import Evo2Tokenizer, register_evo2_tokenizer
register_evo2_tokenizer()


class DNALLMModel(nn.Module):
    """
    vLLM-backed DNA-LLM for inference with prompt_embeds.

    - Text backbone served by vLLM (no HF forward pass).
    - DNA encoder can be any HF MaskedLM (e.g., NucleotideTransformer) or Evo2.
    - DNA token embeddings are projected to the text hidden size and injected
      by replacing <|dna_pad|> placeholder tokens in the input sequence.
    """

    def __init__(
        self,
        ckpt_dir: str,
        text_model_name: str = "Qwen/Qwen3-4B",
        dna_model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        max_length_dna: int = 2048,
        max_length_text: int = 4096,
        text_model_finetune: bool = False,
        dna_model_finetune: bool = False,
        dna_is_evo2: bool = False,
        dna_embedding_layer: Optional[str] = None,
        gpu_memory_utilization: float = 0.4,
        max_model_len: int = 32768,
    ):
        """
        Args:
            ckpt_dir: Local directory of the text LLM (with safetensors) or a model ID resolvable by vLLM.
            text_model_name: Used only to select the chat template (you can set it equal to ckpt_dir).
            dna_model_name: HF MaskedLM (e.g., NucleotideTransformer) or Evo2 name/path if dna_is_evo2=True.
            cache_dir: HF cache directory.
            max_length_dna: Maximum DNA length per sequence.
            max_length_text: Max input tokens (pre-embedding replacement) you plan to feed.
            text_model_finetune: Unused by vLLM (kept for interface parity).
            dna_model_finetune: If True and using HF encoder, gradients would be enabled (but we eval() by default).
            dna_is_evo2: Use Evo2 backend for DNA embeddings.
            dna_embedding_layer: Evo2 layer name to extract (required when dna_is_evo2=True).
            gpu_memory_utilization: vLLM GPU memory fraction.
            max_model_len: vLLM max model len.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.dna_model_finetune = dna_model_finetune
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer

        self.max_length_dna = max_length_dna
        self.max_length_text = max_length_text
        self.max_model_len = max_model_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Load the text model and tokenizer
        self.text_model = LLM(
            model=ckpt_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prompt_embeds=True,
            trust_remote_code=True,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
        )

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir,
            trust_remote_code=True
        )
        self.text_config = AutoConfig.from_pretrained(
            ckpt_dir,
            trust_remote_code=True
        )

        # Chat template + pad token
        self.text_tokenizer.chat_template = CHAT_TEMPLATE
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        # Add DNA special tokens
        new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.dna_token_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")

        # Initialize local embedding layer
        self._embedding_layer = nn.Embedding(self.text_config.vocab_size, self.text_config.hidden_size).to(
            self.device, dtype=self.dtype
        )
        print(
            f"🧬 Initialized local embedding layer with shape: "
            f"({self.text_config.vocab_size}, {self.text_config.hidden_size})"
        )

        # Load embedding weights from checkpoint
        safetensors_path = os.path.join(ckpt_dir, "model.safetensors")
        if os.path.exists(safetensors_path):
            with safe_open(safetensors_path, framework="pt", device=str(self.device)) as f:
                embed_weights = f.get_tensor("model.embed_tokens.weight")
                self._embedding_layer.weight.data = embed_weights.to(dtype=self.dtype)
                print(f"✅ Loaded embedding weights from '{safetensors_path}'")
        else:
            # Try sharded checkpoints
            shard_files = sorted(glob.glob(os.path.join(ckpt_dir, "model-*.safetensors")))
            if not shard_files:
                raise FileNotFoundError(f"No safetensors files found in {ckpt_dir}")

            print(f"🧬 Found {len(shard_files)} sharded checkpoint files.")
            loaded = False
            for shard_file in shard_files:
                with safe_open(shard_file, framework="pt", device=str(self.device)) as f:
                    if "model.embed_tokens.weight" in f.keys():
                        embed_weights = f.get_tensor("model.embed_tokens.weight")
                        self._embedding_layer.weight.data = embed_weights.to(dtype=self.dtype)
                        print(f"✅ Loaded embedding weights from shard: '{shard_file}'")
                        loaded = True
                        break
            if not loaded:
                raise RuntimeError("Embedding weights not found in any shard.")

        self._embedding_layer = self._embedding_layer.to(self.device, dtype=self.dtype)
        print(f"🧬 Moved embedding layer to device: {self.device}")

        # Load DNA encoder and tokenizer
        if dna_is_evo2:
            if dna_model_name is None:
                raise ValueError("dna_model_name must be provided when dna_is_evo2=True")
            if dna_embedding_layer is None:
                raise ValueError("dna_embedding_layer is required for Evo2 to select which layer to extract.")
            from evo2 import Evo2

            self.dna_model = Evo2(dna_model_name)
            self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
            self.dna_hidden_size = self.dna_model.model.config.hidden_size
            self.dna_model.model = self.dna_model.model.to(self.device, dtype=self.dtype)
        else:
            if dna_model_name is None:
                raise ValueError("dna_model_name must be provided for HF DNA encoders")
            self.dna_model = AutoModelForMaskedLM.from_pretrained(
                dna_model_name, cache_dir=cache_dir, trust_remote_code=True
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
            self.dna_hidden_size = self.dna_model.config.hidden_size
            self.dna_model = self.dna_model.to(self.device, dtype=self.dtype)

        self.text_hidden_size = self.text_config.hidden_size

        # Create projection layer to map DNA embeddings to text model's embedding space
        # Using single Linear layer to match training architecture in dna_llm.py
        self.dna_projection = nn.Linear(self.dna_hidden_size, self.text_hidden_size).to(
            device=self.device, dtype=self.dtype
        )

        # Load custom components (projection weights)
        self.load_custom_components(ckpt_dir)

        # Set models to eval mode
        self._setup_default_eval_mode()

        # Create processor for handling inputs
        self.processor = DLProcessor(tokenizer=self.text_tokenizer, dna_tokenizer=self.dna_tokenizer)

    def _setup_default_eval_mode(self):
        """
        Set all model components to eval mode with frozen parameters by default.
        Note: Text model parameter freezing is skipped since vLLM handles it.
        """
        # DNA encoder
        if self.dna_is_evo2:
            # For Evo2, access the internal model
            self.dna_model.model.eval()
            for p in self.dna_model.model.parameters():
                p.requires_grad = False
        else:
            # For HF models, access directly
            self.dna_model.eval()
            for p in self.dna_model.parameters():
                p.requires_grad = False

        # DNA projection
        self.dna_projection.eval()
        for p in self.dna_projection.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def process_dna_embeddings(
        self,
        dna_tokenized: Dict[str, torch.Tensor],
        batch_idx_map: List[int],
        batch_size: int,
    ) -> List[torch.Tensor]:
        """
        Returns a list of length `batch_size`. Each item is a (T_i, H_text) tensor containing
        the concatenated (projected) DNA embeddings for that batch sample.
        """
        if self.dna_is_evo2:
            if self.dna_embedding_layer is None:
                raise ValueError("dna_embedding_layer must be set for Evo2.")
            # Evo2 path: call with return_embeddings=True and layer_names=[...]
            hidden_states_list = []
            seq_count = dna_tokenized["input_ids"].shape[0]
            for seq_idx in range(seq_count):
                input_ids = dna_tokenized["input_ids"][seq_idx : seq_idx + 1]
                _, embeddings = self.dna_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[self.dna_embedding_layer],
                )
                seq_embeddings = embeddings[self.dna_embedding_layer].squeeze(0)  # (L, H_dna)
                hidden_states_list.append(seq_embeddings)
            hidden_states = torch.stack(hidden_states_list) if hidden_states_list else torch.empty(0)
        else:
            outputs = self.dna_model(
                input_ids=dna_tokenized["input_ids"],
                attention_mask=dna_tokenized["attention_mask"],
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]  # (N, L, H_dna)

        # Project to text space
        hidden_states = hidden_states.to(
            device=self.dna_projection.weight.device, dtype=self.dna_projection.weight.dtype
        )
        projected = self.dna_projection(hidden_states)  # (N, L, H_text)

        # Group per batch item
        per_batch: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        for seq_idx, b_idx in enumerate(batch_idx_map):
            valid_len = int(dna_tokenized["attention_mask"][seq_idx].sum().item())
            seq_embed = projected[seq_idx, :valid_len]  # (valid_len, H_text)
            per_batch[b_idx].append(seq_embed)

        out: List[torch.Tensor] = []
        for i in range(batch_size):
            if per_batch[i]:
                out.append(torch.cat(per_batch[i], dim=0))
            else:
                out.append(torch.zeros((0, self.text_hidden_size), device=projected.device, dtype=projected.dtype))
        return out

    # HF forward path is disabled; use .generate()
    def forward(self, *args, **kwargs):
        raise RuntimeError("HF forward is not supported with vLLM. Use .generate() with prompt_embeds.")

    # vLLM generation with prompt_embeds
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        **generation_kwargs: Any,
    ) -> List[str]:
        """
        Replace <|dna_pad|> token positions with projected DNA embeddings, then call vLLM.

        Args:
            input_ids: (B, T) token ids (already templated and containing <|dna_pad|> placeholders)
            attention_mask: (B, T) mask (unused by vLLM; kept for parity)
            dna_tokenized: Tokenized DNA inputs (N sequences) aligned with batch_idx_map
            batch_idx_map: List of length N mapping each DNA sequence to its batch item [0..B-1]
            generation_kwargs:
                temperature, top_p, max_new_tokens, stop (list of strings), etc.

        Returns:
            List[str] decoded generations (length B).
        """
        if input_ids is None or attention_mask is None:
            raise ValueError("input_ids and attention_mask must be provided")

        batch_size = input_ids.shape[0]

        # 1) Get text token embeddings locally
        input_ids = input_ids.to(self.device)
        text_inputs_embeds = self._embedding_layer(input_ids)  # (B, T, H)

        # 2) Inject DNA embeddings where <|dna_pad|> appears
        if dna_tokenized is not None and batch_idx_map is not None:
            dna_tokenized = {k: v.to(self.device) for k, v in dna_tokenized.items()}
            batch_dna_embeds = self.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)

            mask = (input_ids == self.dna_token_id)  # (B, T)
            n_dna_tokens = int(mask.sum().item())

            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)  # (sum_T_i, H)
            n_dna_features = int(dna_embeds_flat.shape[0])

            if n_dna_features != n_dna_tokens:
                raise ValueError(
                    f"DNA features and DNA tokens do not match: features {n_dna_features}, tokens {n_dna_tokens}"
                )

            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype, device=text_inputs_embeds.device)
            text_inputs_embeds[mask] = dna_embeds_flat

        # 3) Build vLLM SamplingParams
        sampling_params = SamplingParams(
            temperature=generation_kwargs.get("temperature", 0.7),
            top_p=generation_kwargs.get("top_p", 0.9),
            max_tokens=generation_kwargs.get("max_new_tokens", 256),
            stop=generation_kwargs.get("stop", ["<|im_end|>"]),
        )

        # 4) Construct per-sample requests with prompt_embeds
        requests = [{"prompt_embeds": text_inputs_embeds[i]} for i in range(batch_size)]

        # 5) Generate
        vllm_outputs = self.text_model.generate(requests, sampling_params=sampling_params)

        # 6) Collect strings
        return [out.outputs[0].text for out in vllm_outputs]

    def load_custom_components(self, llm_dir: str) -> None:
        """
        Load trained dna_projection. DNA encoder is already loaded from HuggingFace in __init__.
        """
        # DNA projection
        projection_path = os.path.join(llm_dir, "dna_projection.pt")
        if os.path.exists(projection_path):
            state = torch.load(projection_path, map_location=self.device)
            self.dna_projection.load_state_dict(state, strict=True)
            self.dna_projection = self.dna_projection.to(device=self.device, dtype=self.dtype)
            print(f"✅ Loaded DNA projection from {projection_path}")
        else:
            raise FileNotFoundError(f"DNA projection not found at {projection_path}")
