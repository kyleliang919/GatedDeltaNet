# Gated Delta Networks: Improving Mamba2 with Delta Rule

![nvidia-deltanet-badge](https://github.com/user-attachments/assets/35b89293-29e9-4560-864d-45f702a5ddf7)

Official PyTorch implementation of [**Gated Delta Networks: Improving Mamba2 with Delta Rule (ICLR '25)**](https://arxiv.org/abs/2412.06464v1). 

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/GatedDeltaNet.svg?style=social)](https://github.com/NVlabs/GatedDeltaNet/stargazers)

[Songlin Yang](https://sustcsonglin.github.io/),
[Jan Kautz](https://jankautz.com/) and
[Ali Hatamizadeh](https://research.nvidia.com/person/ali-hatamizadeh). 

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

For additional functionalities, such as varlen training and inference support, see [FLA implementation](https://github.com/fla-org/flash-linear-attention/tree/main/fla/models/gated_deltanet).

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 180 32">
  <!-- Background rectangle -->
  <rect width="180" height="32" rx="6" fill="#1a1a1a"/>
  
  <!-- NVIDIA logo style -->
  <text x="10" y="21" font-family="Arial, sans-serif" font-weight="bold" font-size="14" fill="#76B900"></text>
  
  <!-- Divider -->
  <line x1="70" y1="8" x2="70" y2="24" stroke="#333" stroke-width="1"/>
  
</svg>

## 🌟 Why Gated DeltaNet?

Gated DeltaNet introduces a novel approach to linear transformers by combining:
- 🧠 **Smart Memory Management**: Intelligent memory management that knows what to keep and what to forget
- ⚡ **Precise Updates**: Targeted memory updates that enhance model efficiency
- 💻 **Hardware Efficiency**: Optimized implementation for real-world deployment
  
![Architecture Overview](https://github.com/user-attachments/assets/70f96a7e-e51d-4514-a429-2ae30c52afbb)


### Efficiency
Gated DeltaNet shows exceptional performance in terms of training throughput compared to models like Mamba2 and Samba:

<p align="center">
<img src="https://github.com/user-attachments/assets/b5c96369-a998-442b-ad7c-2f9fb6979b44" width=62% height=62% 
class="center">
</p>


### Language Modeling and Reasoning

Our model outperforms competitors of various types(e.g. Transformer, RNN, hybrid) in terms of perplexity and zero-shot accuracy on reasoning benchmarks:  

<p align="center">
<img src="https://github.com/user-attachments/assets/afaa4527-e974-4367-a784-6e19c21c8bc0" width=82% height=82% 
class="center">
</p>


### Long-context

Gated DeltaNet also achieves favorable perplexity scores on long-context benchmarks:

<p align="center">
<img src="https://github.com/user-attachments/assets/64c307f4-3b30-4899-ab17-6507e6506c51" width=72% height=72% 
class="center">
</p>


## 📢 Latest Updates
- `02/22/2024`: 🔥🔥 We have introduced new [kernels](https://github.com/NVlabs/GatedDeltaNet/blob/main/lit_gpt/gated_delta_rule_ops/chunk.py) that significantly improve speed ! 
- `02/22/2024`: 🔥 Gated DeltaNet is now avaiable in [FLA](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule) !
- `01/22/2024`: 🔥🔥 Gated DeltaNet has been accepted to ICLR '25.
- `12/09/2024`: **Code Release**: Train your own Gated DeltaNet on Slimpajama dataset
- Watch this space for more exciting updates!

## 🚀 Getting Started

### Training Your Model

Launch your training with our streamlined command:

```bash
python ../pretrain.py \
--train_data_dir ${TRAIN_DATA} \
--val_data_dir ${VALIDATION_DATA} \
--output_root ${SAVE_DIR} \
--exp_name ${NAME} \
--model_name ${MODEL} \
--train_config ${CONFIG} \
--eval_iters ${EVAL_ITERS} \
--learning_rate ${LR} \
--micro_batch_size ${MICRO_BATCH_SIZE}
```
💡 **Pro Tip**: Add `--interactive_job --debug` for interactive debugging sessions!

Please see this slurm [script](https://github.com/NVlabs/GatedDeltaNet/blob/main/scripts/tsz512x4k_15B_gated_deltanet_h1_0.4B.sh) for training the GatedDeltaNet_H1 model with 0.4B parameters on 15B tokens. The training requires 4 nodes and can be finished in approximately 4 hours. For this run, the validation loss and perplexitty curves (1x & 2x for lengh extrapolation) are expected as follows:

![curves](https://github.com/user-attachments/assets/bd8afd42-6f20-4103-8b31-48516871b681)


## 📜 License

Copyright © 2024, NVIDIA Corporation. All rights reserved.

Licensed under the NVIDIA Source Code License-NC. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

Built on the shoulders of giants:
- [Samba](https://github.com/microsoft/Samba)
- [LiTGPT](https://github.com/Lightning-AI/litgpt)
- [TinyLLaMa](https://github.com/jzhang38/TinyLlama)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)

## ⭐ Support Us

If you find this work useful, please consider:
- Starring the repository
- Citing our paper
- Contributing to the codebase

Join us in pushing the boundaries of linear transformers! 🚀

## Citation

If you find Gated DeltaNet to be useful for your work, please consider citing our paper: 

```
@inproceedings{yang2025gated,
title={Gated Delta Networks: Improving Mamba2 with Delta Rule},
author={Songlin Yang and Jan Kautz and Ali Hatamizadeh},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=r8H7xhYPwz}
}
```

## Star History

[![Stargazers repo roster for @NVlabs/GatedDeltaNet](https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=NVlabs&repo=GatedDeltaNet)](https://github.com/NVlabs/GatedDeltaNet/stargazers)


[![Star History Chart](https://api.star-history.com/svg?repos=NVlabs/GatedDeltaNet&type=Date)](https://star-history.com/#NVlabs/GatedDeltaNet&Date)
