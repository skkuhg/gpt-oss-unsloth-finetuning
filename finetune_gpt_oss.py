#!/usr/bin/env python3
"""
GPT-OSS 20B Fine-tuning with Unsloth

This script demonstrates how to fine-tune OpenAI's GPT-OSS 20B model using Unsloth
for 2x faster training and reduced memory usage.

Author: Fine-tuned with Unsloth (https://unsloth.ai)
"""

import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import standardize_sharegpt

def main():
    print("🚀 Starting GPT-OSS 20B fine-tuning with Unsloth...")
    
    # Configuration
    max_seq_length = 4096
    dtype = None
    model_name = "unsloth/gpt-oss-20b"
    
    print(f"📦 Loading model: {model_name}")
    
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=dtype,  # None for auto detection
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # 4-bit quantization to reduce memory
        full_finetuning=False,  # Use LoRA for efficiency
        # token="hf_...",  # Use if accessing gated models
    )
    
    print("🔧 Adding LoRA adapters for parameter-efficient fine-tuning...")
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Optimized for Unsloth
        bias="none",     # Optimized for Unsloth
        use_gradient_checkpointing="unsloth",  # 30% less VRAM
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("📊 Loading and preparing dataset...")
    
    # Load dataset
    dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split="train")
    
    # Format dataset
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}
    
    # Standardize and format dataset
    dataset = standardize_sharegpt(dataset)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    
    print("🏋️ Setting up trainer...")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,  # Set num_train_epochs=1 for full training
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # Change to "wandb" for logging
        ),
    )
    
    # Show memory stats before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"🖥️  GPU: {gpu_stats.name}. Max memory: {max_memory} GB.")
    print(f"💾 {start_gpu_memory} GB of memory reserved.")
    
    print("🚂 Starting training...")
    
    # Train the model
    trainer_stats = trainer.train()
    
    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print("\n📈 Training completed!")
    print(f"⏱️  Training time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"⏰ Training time: {trainer_stats.metrics['train_runtime']/60:.2f} minutes")
    print(f"💾 Peak reserved memory: {used_memory} GB")
    print(f"🧠 Peak reserved memory for training: {used_memory_for_lora} GB")
    print(f"📊 Peak reserved memory %: {used_percentage}%")
    print(f"🎯 Peak reserved memory for training %: {lora_percentage}%")
    
    print("\n🧪 Testing inference with different reasoning efforts...")
    
    # Test inference with different reasoning efforts
    test_inference(model, tokenizer, reasoning_effort="low")
    test_inference(model, tokenizer, reasoning_effort="medium")
    test_inference(model, tokenizer, reasoning_effort="high")
    
    print("✅ Fine-tuning completed successfully!")

def test_inference(model, tokenizer, reasoning_effort="medium"):
    """Test inference with specified reasoning effort level"""
    print(f"\n🔍 Testing with reasoning effort: {reasoning_effort}")
    
    messages = [
        {
            "role": "system", 
            "content": "reasoning language: French\n\nYou are a helpful assistant that can solve mathematical problems."
        },
        {"role": "user", "content": "Solve x^5 + 3x^4 - 10 = 3."},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        reasoning_effort=reasoning_effort,
    ).to(model.device)
    
    # Adjust max tokens based on reasoning effort
    max_tokens = {
        "low": 512,
        "medium": 1024,
        "high": 2048
    }
    
    print(f"🎭 Generating response with {max_tokens[reasoning_effort]} max tokens...")
    _ = model.generate(
        **inputs, 
        max_new_tokens=max_tokens[reasoning_effort], 
        streamer=TextStreamer(tokenizer)
    )

if __name__ == "__main__":
    main()