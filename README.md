# 🔮 Gated Delta Networks: Improving Mamba2 with Delta Rule

Official PyTorch implementation of Gated Delta Networks: Improving Mamba2 with Delta Rule.

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-NVIDIA%20Source%20Code%20License--NC-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2312.XXXXX-b31b1b.svg)](https://arxiv.org)

[Songlin Yang](https://sustcsonglin.github.io/),
[Jan Kautz](https://jankautz.com/) and
[Ali Hatamizadeh](https://research.nvidia.com/person/ali-hatamizadeh). 


## 🌟 Why Gated DeltaNet?

Gated DeltaNet introduces a groundbreaking approach to linear transformers by combining:
- 🧠 **Smart Memory Clearing**: Intelligent memory management that knows what to keep and what to forget
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
<img src="https://github.com/user-attachments/assets/64c307f4-3b30-4899-ab17-6507e6506c51" width=62% height=62% 
class="center">
</p>


## 📢 Latest Updates

- `12/09/2024`: 🔥 **Code Release**: Train your own Gated DeltaNet on Slimpajama dataset
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

## 📜 License

Copyright © 2024, NVIDIA Corporation. All rights reserved.

Licensed under the NVIDIA Source Code License-NC. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

Built on the shoulders of giants:
- [Samba](https://github.com/microsoft/Samba)
- [LiTGPT](https://github.com/Lightning-AI/litgpt)

## ⭐ Support Us

If you find this work useful, please consider:
- Starring the repository
- Citing our paper
- Contributing to the codebase

Join us in pushing the boundaries of linear transformers! 🚀

## Star History

[![Stargazers repo roster for @NVlabs/GatedDeltaNet](https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=NVlabs&repo=GatedDeltaNet)](https://github.com/NVlabs/GatedDeltaNet/stargazers)


[![Star History Chart](https://api.star-history.com/svg?repos=NVlabs/GatedDeltaNet&type=Date)](https://star-history.com/#NVlabs/GatedDeltaNet&Date)
