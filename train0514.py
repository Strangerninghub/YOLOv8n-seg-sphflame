import os
import yaml
from ultralytics import YOLO
import torch
torch.backends.cudnn.enabled = False
import gc


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def verify_annotations():
    """验证标注文件，确保只包含flame类的分割标注"""
    labels_dir = 'datasets/segment/seg/labels'

    for split in ['train', 'val']:
        split_dir = os.path.join(labels_dir, split)
        if not os.path.exists(split_dir):
            continue

        for label_file in os.listdir(split_dir):
            if label_file.endswith('.txt'):
                label_path = os.path.join(split_dir, label_file)
                new_lines = []

                with open(label_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == '0':  # flame类
                        new_lines.append(line)

                with open(label_path, 'w') as f:
                    f.writelines(new_lines)


def modify_existing_config():
    """修改现有的segment.yaml配置文件"""
    original_config_path = 'datasets/segment/seg/segment.yaml'

    with open(original_config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['nc'] = 1
    config['names'] = ['flame']

    modified_config_path = 'flame_segmentation_modified.yaml'
    with open(modified_config_path, 'w') as f:
        yaml.dump(config, f)

    return modified_config_path


def train_flame_segmentation_optimized():
    """优化的训练函数，从高配置开始尝试，失败则自动降级"""

    clear_memory()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # GPU信息
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # ===== 将 weights/ 中的 yolo11n.pt 复制到当前目录，供 AMP 检查使用 =====
        import shutil

        local_weights = 'yolo11n.pt'  # 当前目录下的文件名
        source_weights = os.path.join('weights', 'yolo11n.pt')  # 您的存放路径

        if not os.path.exists(source_weights):
            raise FileNotFoundError(f"权重文件未找到: {source_weights}，请从官方下载 yolo11n.pt 放入 weights/")

        # 如果当前目录已有，先删除再复制，保证使用的是 weights/ 中的版本
        if os.path.exists(local_weights):
            os.remove(local_weights)
        shutil.copy2(source_weights, local_weights)

        # 快速验证文件是否为有效的PyTorch模型（避免损坏的zip导致同样错误）
        try:
            _ = torch.load(local_weights, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"{local_weights} 文件已损坏，请重新下载官方 yolo11n.pt 放入 weights/") from e

        print(f"已从 {source_weights} 复制 yolo11n.pt 到当前目录，并验证通过。")
        # ===== 复制结束 =====

        verify_annotations()
        data_config = modify_existing_config()

    # 配置尝试顺序：从高资源需求到低资源需求，失败则降级
    configs = [
        {
            'name': 'ultra',
            'model': 'YOLOv8n-seg-sphflame-v1.0.pt',
            'imgsz': 768,
            'batch': 16,
            'epochs': 250,
            'lr0': 0.001,
            'patience': 30,
            'mosaic': 1.0,
            'warmup_epochs': 4
        },
        {
            'name': 'medium',
            'model': 'YOLOv8n-seg-sphflame-v1.0.pt',
            'imgsz': 768,
            'batch': 8,
            'epochs': 200,
            'lr0': 0.0005,
            'patience': 30,
            'mosaic': 1.0,
            'warmup_epochs': 3
        },
        {
            'name': 'low',
            'model': 'YOLOv8n-seg-sphflame-v1.0.pt',
            'imgsz': 640,
            'batch': 4,
            'epochs': 200,
            'lr0': 0.00025,
            'patience': 20,
            'mosaic': 0.8,
            'warmup_epochs': 3
        },
        {
            'name': 'minimal',          # 极限低配，几乎不可能爆显存
            'model': 'YOLOv8n-seg-sphflame-v1.0.pt',
            'imgsz': 416,
            'batch': 2,
            'epochs': 100,
            'lr0': 0.000125,
            'patience': 20,
            'mosaic': 0.5,
            'warmup_epochs': 2
        }
    ]

    for config in configs:
        print(f"\n尝试配置: {config['name']}")
        print(f"模型: {config['model']}, 图像尺寸: {config['imgsz']}, 批次: {config['batch']}, 学习率: {config['lr0']}")

        try:
            clear_memory()
            model = YOLO(config['model'])

            train_args = {
                'data': data_config,
                'epochs': config['epochs'],
                'imgsz': config['imgsz'],
                'batch': config['batch'],
                'device': device,
                'workers': 4,
                'patience': config['patience'],
                'save': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': config['lr0'],
                'cos_lr': True,
                'amp': True,
                'project': 'flame_segmentation',
                'name': f'flame_seg_{config["name"]}',
                # 数据增强参数（恢复颜色/几何扰动，注意非3通道图片可能自动转为3通道）
                'warmup_epochs': config['warmup_epochs'],
                'mosaic': config['mosaic'],
                'mixup': 0.0,
                'copy_paste': 0.0,
                'hsv_h': 0.015,         # 色调扰动 ±0.015
                'hsv_s': 0.7,           # 饱和度扰动 ±70%
                'hsv_v': 0.4,           # 明度扰动 ±40%
                'degrees': 10.0,        # 旋转 ±10°
                'translate': 0.1,       # 平移 ±10%
                'scale': 0.5,           # 缩放 ±50%
                'shear': 2.0,           # 剪切 ±2°
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,          # 水平翻转概率 50%
                'erasing': 0.0,
            }

            results = model.train(**train_args)
            print(f"配置 {config['name']} 训练成功!")
            return results

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"配置 {config['name']} 内存不足，尝试下一个配置...")
                continue
            else:
                raise e

    print("所有配置都内存不足，建议:")
    print("1. 进一步降低图像尺寸到320x320（修改 minimal 配置）")
    print("2. 使用CPU训练（速度较慢）")
    print("3. 使用Google Colab等云端GPU")


if __name__ == "__main__":
    train_flame_segmentation_optimized()
