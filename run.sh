#训练模型
# --amp 是启动训练时启用混合精度训练（Mixed Precision Training） 的命令行参数
python train.py --amp

#训练模型（禁用wandb）
python train_disable_wandb.py --amp