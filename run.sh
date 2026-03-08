#训练模型
# --amp 是启动训练时启用混合精度训练（Mixed Precision Training） 的命令行参数
python train.py --amp

#训练模型（禁用wandb）
python train_disable_wandb.py --amp

#测试模型
#测试并保存结果
python predict.py --model ./checkpoints/checkpoint_epoch5.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" -o output.jpg
#测试不保存结果（仅展示）
python predict.py --model ./checkpoints/checkpoint_epoch5.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" --viz --no-save