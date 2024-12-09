# Gated Delta Networks: Improving Mamba2 with Delta Rule

Official PyTorch implementation of Gated Delta Networks: Improving Mamba2 with Delta Rule.

[Songlin Yang](https://sustcsonglin.github.io/),
[Jan Kautz](https://jankautz.com/) and
[Ali Hatamizadeh](https://research.nvidia.com/person/ali-hatamizadeh). 


Gated DeltaNet is a novel LLM architecture that advances the capabilities of linear transformers. We introduce gated delta rule which effectively combines memory clearing with precise and targeted updates in a hardware-efficient implementation.



Gated DeltaNet demonstrates a strong performance on various LLM benchmarks such as common-sense reasoning and recall. It also achieves favorable training throughput compared to competeting models such as Mamba2 and Samba. 

### Training

We provide a sample script for running slurm job with GatedDeltaNet_H1_0.4B model in scripts folder. In general, a training job can be initiated using the following command:

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
--micro_batch_size ${MICRO_BATCH_SIZE} \
```

You can run the job interactively for debugging purposes by passing arguments such as ```--interactive_job --debug```.

## Licenses

Copyright © 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.


## Acknowledgement
This repository is built on top of the [Samba](https://github.com/microsoft/Samba) and [LiTGPT](https://github.com/Lightning-AI/litgpt) repositories.