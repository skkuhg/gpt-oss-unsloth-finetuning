# GPT-OSS (20B) Fine-tuning with Unsloth

🚀 **Fine-tune OpenAI's GPT-OSS 20B model with Unsloth for 2x faster training and reduced memory usage!**

This repository demonstrates how to fine-tune the new OpenAI GPT-OSS 20B model using the powerful [Unsloth](https://unsloth.ai/) framework. Unsloth enables efficient fine-tuning with significant speed improvements and memory optimization.

## 🌟 Features

- **2x Faster Training**: Leverages Unsloth's optimized kernels for accelerated fine-tuning
- **Memory Efficient**: Uses 4-bit quantization and LoRA adapters to reduce VRAM usage
- **Reasoning Effort Control**: Demonstrates GPT-OSS's unique reasoning effort levels (low, medium, high)
- **Multilingual Support**: Fine-tuned on multilingual reasoning datasets
- **Production Ready**: Includes complete training pipeline and inference examples

## 🔧 Key Technologies

- **[Unsloth](https://github.com/unslothai/unsloth)**: Ultra-fast LLM fine-tuning framework
- **GPT-OSS 20B**: OpenAI's open-source reasoning model
- **LoRA**: Parameter-efficient fine-tuning technique
- **4-bit Quantization**: Memory optimization using bitsandbytes

## 📋 Requirements

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/gpt-oss-unsloth-finetuning.git
cd gpt-oss-unsloth-finetuning
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the fine-tuning script**:
```bash
python finetune_gpt_oss.py
```

## 🎯 Model Capabilities

### Reasoning Effort Levels
GPT-OSS models include a unique "reasoning effort" feature that controls the trade-off between performance and response speed:

- **Low**: Fast responses for simple tasks
- **Medium**: Balanced performance and speed
- **High**: Maximum reasoning performance for complex tasks

### Training Dataset
This implementation uses the `HuggingFaceH4/Multilingual-Thinking` dataset, which contains:
- Chain-of-thought reasoning examples
- Multilingual translations (4 languages)
- Complex mathematical problem solving

## 📊 Performance

- **Memory Usage**: ~6GB VRAM with 4-bit quantization
- **Training Speed**: 2x faster than standard implementations
- **Model Size**: 20B parameters with LoRA adapters (1% trainable parameters)

## 🔬 Technical Details

### Model Configuration
- **Base Model**: `unsloth/gpt-oss-20b`
- **Quantization**: 4-bit using bitsandbytes
- **LoRA Rank**: 8
- **Target Modules**: All attention and MLP layers
- **Context Length**: 4096 tokens

### Training Configuration
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **Optimizer**: AdamW 8-bit
- **Training Steps**: 60 (configurable)

## 🤝 Acknowledgments

Special thanks to the [Unsloth team](https://unsloth.ai/) for their incredible work on making LLM fine-tuning accessible and efficient. This project would not be possible without their innovative framework.

- **Unsloth GitHub**: https://github.com/unslothai/unsloth
- **Unsloth Documentation**: https://docs.unsloth.ai/
- **Unsloth Discord**: https://discord.gg/unsloth

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🛠️ Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📚 Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [OpenAI GPT-OSS Cookbook](https://cookbook.openai.com/articles/gpt-oss/fine-tune-transfomers)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl)

## ⭐ Star History

If this repository helps you, please consider giving it a star! ⭐

---

**Built with ❤️ using [Unsloth](https://unsloth.ai/)**