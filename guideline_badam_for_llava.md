Please refer to the changes in the latest commit for using BAdam. In particular, i have made the following modifications:

1. Wrap the original optimizer in function `create_optimizer` and specify BAdam's parameters in **llava_trainer.py**. Important parameters are
    * `switch_block_every`: The $K$ in paper. Values around $100$ usually gives good convergence.
    * `block_prefix_list`: Block partition strategy; see [here](https://github.com/Ledzy/BAdam?tab=readme-ov-file#partition-by-module-a-single-gpu) on detailed instruction on its format. When training Llava, we observe that joint training of LLM layer and vision encoder layers yields faster convergence compared with only train the LLM layer. Since the size of vision encoder is usually small, we suggest to keep it trainable through all the time.

2. Change the gradient_checkpointing_enable into a customized one for avoiding unnecessary BP cost and save training time in **llava_trainer.py**. (However, when the vision encoder is trainable, we need to BP to the LLM's first layer, so this operation doesn't save time anymore)

3. Add `BAdamCallback` when initializing Trainer in **train.py**. This is used for set up optimizer when using Deepspeed ZeRO-3.

4. Comment the @torch.no_grad() decorator for the `forward` of the vision model to enable the training of vision encoder in **clip_encoder.py**

Please see `requirements.txt` for the environment that i used for conducting experiments, especially the version for deepspeed, accelerate, and badam.