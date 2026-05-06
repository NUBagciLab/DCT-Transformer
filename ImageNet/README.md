These experiments are based on [PyTorch's official ImageNet training](https://github.com/pytorch/vision/tree/main/references/classification#swintransformer).

Folders `swin-t` and `vit_b_32` are the vanilla transformers for reference. All other folders were revised based on them.

Folders like `swin_t_dctk` and `swin_t_dctk_trainable` are the DCT-initialized transformers, where with `_trainable` means the DCT weights are trainable (proposed implementation), and without `_trainable` means the DCT weights are fixed (for ablation study). `q, k, v` after `dct` represents which weights were initialized as the DCT matrices, such as `dctk` means the weights to compute keys (K) were initialized as the DCT matrices. You can change `dct_init='k'`

    self.self_attention = MultiheadAttention_DCT_Init(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True, dct_init='k')

at `line 104` in `vit_b_32_dctk_trainable` if you want to try other DCT initialization.

Folders `swin_t_DCT-Former0.75` and `vit_b_32_DCT-Former0.75` are the DCT-compressed transformers. The default compression ratio is `0.75`. If you want to try other compression ratios (`0.25` or `0.5`), you can change 

    dct_ratio: float = 0.75

at `line 252` in `swin_transformer.py` and `line 98` in `vision_transformer.py`, respectively.

To run training:

### SwinTransformer
```
torchrun --nproc_per_node=8 train.py\ 
--model $MODEL --epochs 300 --batch-size 128 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs 20 --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224
```
Here `$MODEL` is one of `swin_t`, `swin_s` or `swin_b`.
Note that `--val-resize-size` was optimized in a post-training step, see their `Weights` entry for the exact value.

#### vit_b_32
```
torchrun --nproc_per_node=8 train.py\
    --model vit_b_32 --epochs 300 --batch-size 512 --opt adamw --lr 0.003 --wd 0.3\
    --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 30\
    --lr-warmup-decay 0.033 --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment imagenet\
    --clip-grad-norm 1 --ra-sampler --cutmix-alpha 1.0 --model-ema
```

Note that the above command corresponds to training on a single node with 8 GPUs.
For generating the pre-trained weights, we trained with 2 nodes, each with 8 GPUs (for a total of 16 GPUs),
and `--batch_size 256`.
