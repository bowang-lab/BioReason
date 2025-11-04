#!/usr/bin/env python
"""
Test script for DNA-vLLM model inference.
Tests the vLLM-backed DNA-LLM model with a simple text-only question.
"""

import os
import sys
from bioreason.models.dna_vllm import DNALLMModel

def main():
    print("=" * 80)
    print("DNA-vLLM Test Script")
    print("=" * 80)
    
    # Configuration
    ckpt_dir = "/large_storage/goodarzilab/bioreason/checkpoints/nt-500m-qwen3-4b-finetune-kegg-Qwen3-4B-20250511-190543/nt-500m-qwen3-4b-finetune-kegg-Qwen3-4B-epoch=03-val_loss_epoch=0.3599.ckpt/output_dir"
    text_model_name = "Qwen/Qwen3-4B"
    dna_model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    cache_dir = "/large_storage/goodarzilab/bioreason/cache_dir"
    
    print(f"\n📁 Checkpoint directory: {ckpt_dir}")
    print(f"🤖 Text model: {text_model_name}")
    print(f"🧬 DNA model: {dna_model_name}")
    
    # Check if checkpoint directory exists
    if not os.path.exists(ckpt_dir):
        print(f"\n❌ ERROR: Checkpoint directory not found: {ckpt_dir}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Initializing DNA-vLLM Model...")
    print("=" * 80)
    
    try:
        # Initialize the model (using training config values)
        model = DNALLMModel(
            ckpt_dir=ckpt_dir,
            text_model_name=text_model_name,
            dna_model_name=dna_model_name,
            cache_dir=cache_dir,
            max_length_dna=2048,
            max_length_text=512,
            text_model_finetune=False,
            dna_model_finetune=False,
            dna_is_evo2=False,
            dna_embedding_layer=None,
            gpu_memory_utilization=0.4,
            max_model_len=8192,
        )
        print("\n✅ Model initialized successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR during model initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Prepare test input
    print("\n" + "=" * 80)
    print("Preparing Test Input...")
    print("=" * 80)
    
    test_question = "What is the capital of France?"
    print(f"\n📝 Question: {test_question}")
    
    # Format as chat using the processor
    messages = [
        {"role": "user", "content": test_question}
    ]
    
    # Apply chat template
    formatted_text = model.text_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(f"\n💬 Formatted prompt:\n{formatted_text}")
    
    # Tokenize
    inputs = model.text_tokenizer(
        formatted_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=model.max_length_text
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"\n🔢 Input shape: {input_ids.shape}")
    print(f"   Token count: {input_ids.shape[1]}")
    
    # Generate response
    print("\n" + "=" * 80)
    print("Generating Response...")
    print("=" * 80)

    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            dna_tokenized=None,  # No DNA input for this test
            batch_idx_map=None,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=256,
            stop=["<|im_end|>"],
        )
        
        print("\n✅ Generation completed!")
        print("\n" + "=" * 80)
        print("Response:")
        print("=" * 80)
        
        # outputs should be a list of strings
        for i, output in enumerate(outputs):
            print(f"\n[Response {i+1}]")
            print(output)
        
    except Exception as e:
        print(f"\n❌ ERROR during generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Test completed successfully! ✨")
    print("=" * 80)

if __name__ == "__main__":
    main()

