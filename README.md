# Zoom-Transformer

# Dependencies
- Python >= 3.6
- PyTorch >= 1.2.0
- pyyaml, tqdm, tensorboardX

# Train
Dataset: Volleyball-Skeleton-Activity

Download: [VS-A](https://whueducn-my.sharepoint.com/:f:/g/personal/zjiaxu_whu_edu_cn/EpDY5l3v4BJEnUnswFTc8aMBfrtVf8KruGxgwRXnwaFpMg?e=g2pfbm)
```
python train.py --config ./config/train_cfg.yaml
```
# Inference
```
python test.py --config ./config/test_cfg.yaml
```
