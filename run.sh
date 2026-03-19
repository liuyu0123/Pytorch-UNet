#训练模型
# --amp 是启动训练时启用混合精度训练（Mixed Precision Training） 的命令行参数
python train.py --amp

#训练模型（禁用wandb）
python train_disable_wandb.py --amp
#训练模型（禁用wandb），去掉--amp防止训练发散
python train_disable_wandb.py

#训练模型（带参数版本）
# 基础用法
python train_disable_wandb_params.py `
    -i "D:/Files/Data/USVInlandDataset/Water Segmentation/training/training/640_320_undistorted" `
    -m "D:/Files/Data/USVInlandDataset/Water Segmentation/training/training/640_320_undistorted_gif"
# 完整参数示例
python train_disable_wandb_params.py `
    -i "D:/Files/Data/images" `
    -m "D:/Files/Data/masks" `
    --mask-suffix "_mask" `
    -e 10 `
    -b 4 `
    -l 1e-4 `
    --amp

#训练模型（train和val分离版本）
#方式1：自动划分（原脚本行为）
python train_water.py `
    --images data/all/images `
    --masks data/all/masks `
    --validation 10
#方式2：使用预分好的 train/val
python train_water.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_white `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_white `
    --epochs 100 `
    --batch-size 8 `
    --learning-rate 5e-4
#方式3：混合精度 + 多 workers（Linux/Mac）
python train_water.py `
    --images data/train/images `
    --masks data/train/masks `
    --val-images data/val/images `
    --val-masks data/val/masks `
    --amp `
    --workers 4 `
    --epochs 50

#测试模型（测试集）
# 基础测试
python test_water.py `
    --model checkpoints/checkpoint_best.pth `
    --images D:\Files\Data\IRWSB\test\images `
    --masks D:\Files\Data\IRWSB\test\masks_white

# 指定输出路径
python test_water.py `
    --model checkpoints/checkpoint_best.pth `
    --images D:\Files\Data\IRWSB\test\images `
    --masks D:\Files\Data\IRWSB\test\masks_white `
    --output results/experiment1_test.csv `
    --batch-size 4 `
    --amp


#测试模型(单张图片推理)
#测试并保存结果
python predict.py --model ./checkpoints/checkpoint_epoch5.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" -o output.jpg
#测试不保存结果（仅展示）
python predict.py --model ./checkpoints1_trained_gt/checkpoint_epoch1.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" --viz --no-save
python predict.py --model ./checkpoints2_trained_lds/checkpoint_epoch1.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" --viz --no-save
python predict.py --model ./checkpoints3_trained_gif/checkpoint_epoch1.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" --viz --no-save
python predict.py --model ./checkpoints4_trained_gif_noamp/checkpoint_epoch5.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" --viz --no-save
python predict.py --model ./checkpoints/checkpoint_epoch1.pth -i "D:\Files\Data\USVInlandDataset\Water Segmentation\training\training\640_320_undistorted\H05_1_0000000000.jpg" --viz --no-save