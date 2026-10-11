# Paper Implementations

From-scratch PyTorch implementations of six deep learning papers. Each folder has the model built from the paper's
equations (no `nn.Transformer`, `nn.MultiheadAttention`, `torchvision.models` or `peft`), tests for the details that
are easy to get wrong, a small experiment, and a README with results and differences from the paper.

Everything trains on a laptop CPU, so the experiments are small. I was aiming to get the mechanics right and
reproduce the *qualitative* findings, not the papers' absolute numbers.

| Paper | Folder | Experiment | Result |
| --- | --- | --- | --- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | [`transformer/`](transformer) | Encoder-decoder on string reversal | 89.9% exact match, 99.0% per-token accuracy on 2,000 held-out strings |
| [Deep Residual Learning](https://arxiv.org/abs/1512.03385) (He et al., 2015) | [`resnet/`](resnet) | Plain vs. residual nets at 20 and 56 layers on CIFAR-10 | Degradation reproduced: plain-56 plateaus at 72% train error vs. 40% for plain-20; shortcuts bring the 56-layer net to 48% |
| [U-Net](https://arxiv.org/abs/1505.04597) (Ronneberger et al., 2015) | [`unet/`](unet) | Pet vs. background segmentation on Oxford-IIIT Pet (64x64) | Test Dice 0.833, IoU 0.730 |
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2020) | [`vit/`](vit) | Small ViT on FashionMNIST | 87.7% test accuracy with 139k parameters |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020) | [`ddpm/`](ddpm) | Unconditional MNIST generation | Mostly readable digits across all ten classes from a 1M-parameter U-Net |
| [LoRA](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) | [`lora/`](lora) | Adapt an MNIST-pretrained MLP to FashionMNIST | Rank 4 reaches 84.2% vs. 87.1% for full fine-tuning, training 1.7% of the parameters |

<p align="center">
  <img src="ddpm/assets/samples.png" width="32%" alt="DDPM samples">
  <img src="unet/assets/predictions.png" width="64%" alt="U-Net predictions">
</p>
<p align="center">
  <img src="resnet/assets/plain_vs_resnet_20v56.png" width="96%" alt="Plain vs residual training error">
</p>

## Notes

- Transformer tests check that masked positions get exactly zero attention and that the decoder is causal end to end; removing either mask makes them fail.
- The ResNet comparison builds plain and residual nets from the same code, and all four models see identical mini-batches.
- In the paper's configuration the U-Net reproduces Fig. 1 exactly (572x572 input, 388x388 output, 31.0M parameters).
- LoRA accuracy is identical before and after merging the adapter in all 12 runs.

For LoRA on a real language model (GPT-2), see [tinytune](https://github.com/dhruva1311/tinytune).

## Running

Python 3.11 and the packages in `requirements.txt`. No GPU needed.

```bash
pip install -r requirements.txt
cd transformer
python -m pytest test_model.py
python train.py
```

Each folder's README has its exact commands, run times and hyperparameters. Datasets download into each folder's
`data/` and checkpoints go to `checkpoints/`; both are gitignored.

## License

MIT
