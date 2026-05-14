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
    """优化的训练函数，逐步尝试不同配置"""

    clear_memory()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # GPU信息
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    verify_annotations()
    data_config = modify_existing_config()

    # 配置尝试顺序（从低到高资源需求）
    configs = [
        {
            'name': 'ultra',
            'model': 'yolov8n-seg.pt',
            'imgsz': 768,
            'batch': 16,
            'epochs': 200,
            'lr0': 0.002
        },
        {
            'name': 'medium',
            'model': 'yolov8s-seg.pt',
            'imgsz': 768,
            'batch': 8,
            'epochs': 200,
            'lr0': 0.001
        },
        {
            'name': 'low',
            'model': 'yolov8n-seg.pt',
            'imgsz': 640,
            'batch': 4,
            'epochs': 200,
            'lr0': 0.001
        }
    ]

    for config in configs:
        print(f"\n尝试配置: {config['name']}")
        print(f"模型: {config['model']}, 图像尺寸: {config['imgsz']}, 批次: {config['batch']}")

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
                'patience': 20,
                'save': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'lr0': config['lr0'],
                'cos_lr': True,
                'amp': True,
                'project': 'flame_segmentation',
                'name': f'flame_seg_{config["name"]}',
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
    print("1. 进一步降低图像尺寸到256x256")
    print("2. 使用CPU训练（速度较慢）")
    print("3. 使用Google Colab等云端GPU")


if __name__ == "__main__":
    train_flame_segmentation_optimized()