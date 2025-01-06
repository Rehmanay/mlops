import os
import yaml
import argparse
from seg_unet import UNet
from loader import UNETLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
from PIL import Image
from tqdm import tqdm
import torch
from einops import rearrange
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def save_checkpoint(model, train_loss, val_loss, ckpts_dir):
    """Save model checkpoint."""
    os.makedirs(ckpts_dir, exist_ok=True)
    filename = os.path.join(ckpts_dir, f'{str(time.time())[-5:]}.pth')
    save_dict = {
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(save_dict, filename)
    logger.info(f'Checkpoint saved: {filename}')

def helper(dir):
    for dirpath, dirnames, filenames in os.walk(dir):
        print(f"Current directory: {dirpath}")
        print(f"Subdirectories: {dirnames}")
        print(f"Files: {filenames}")

if __name__ == "__main__":
    logger.info("Starting the training script in the SageMaker container")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/opt/ml/model", help="Path to save the trained model")
    parser.add_argument("--output_dir", type=str, default="/opt/ml/output", help="Output directory for logs or other artifacts")
    # Add positional argument for SageMaker's default 'train' argument
    parser.add_argument("mode", nargs='?', default=None, help="SageMaker training mode")
    
    args, unknown = parser.parse_known_args()  # Use parse_known_args instead of parse_args
    
    logger.info(f"Arguments: {args}")
    logger.info(f"Unknown arguments: {unknown}")

    mode = os.environ.get('SM_HP_MODE', 'train')
    reshape_h = int(os.environ.get('SM_HP_RESHAPE_H', 832))
    reshape_w = int(os.environ.get('SM_HP_RESHAPE_W', 544))
    data_path = os.environ.get('SM_CHANNEL_DATA', '/opt/ml/input/data')
    output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
    ckpts_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')

    helper('/opt/ml/')


    unet_init = {
        "num_classes": int(os.environ.get('SM_HP_NUM_CLASSES', 59)),
        "n_channels": int(os.environ.get('SM_HP_N_CHANNELS', 3)),
        "encoder_name": os.environ.get('SM_HP_ENCODER_NAME', 'resnet34'),
        "encoder_weights": os.environ.get('SM_HP_ENCODER_WEIGHTS', 'imagenet')
    }
    train_config = {
        "trainval_split": [float(x) for x in os.environ.get('SM_HP_TRAINVAL_SPLIT', '0.8,0.2').split(',')],
        "epochs": int(os.environ.get('SM_HP_EPOCHS', 1)),
        "batch_size": int(os.environ.get('SM_HP_BATCH_SIZE', 16)),
        "learning_rate": float(os.environ.get('SM_HP_LEARNING_RATE', 0.01)),
        "use_ckpts": os.environ.get('SM_HP_USE_CKPTS', 'False') == 'True',
        "ckpts_file": os.environ.get('SM_HP_CKPTS_FILE', '84527.pth')
    }

    preprocessing = {
        'reshape_h': reshape_h,
        'reshape_w': reshape_w
    }

    model = UNet(**unet_init)
    img_tfs = transforms.Compose([
        Resize((preprocessing['reshape_h'], preprocessing['reshape_w'])),
        ToTensor()
    ])
    mask_tfs = transforms.Compose([
        Resize((preprocessing['reshape_h'], preprocessing['reshape_w'])),
        ToTensor(),
        transforms.Lambda(lambda x: (x * 255).long())
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    logger.info(f"Using device: {device}")
    logger.info(f"data path : {data_path}")

    if mode == "train":
        dataset = UNETLoader(root=data_path, img_tfs=img_tfs, mask_tfs=mask_tfs)
        train_data, val_data = random_split(dataset, train_config['trainval_split'])

        train_loader = DataLoader(train_data, batch_size=train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=train_config['batch_size'], shuffle=True)

        optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        if train_config['use_ckpts']:
            checkpoint_path = os.path.join(ckpts_dir, train_config['ckpts_file'])
            if os.path.exists(checkpoint_path):
                ckpts = torch.load(checkpoint_path, weights_only=True)
                model.load_state_dict(ckpts['model_state_dict'])
                logger.info("Loaded pretrained checkpoint")
            else:
                logger.warning(f"Checkpoint not found at: {checkpoint_path}")
    
        for ep in range(train_config['epochs']):
            logger.info(f"Starting epoch {ep + 1}/{train_config['epochs']}")
            model.train()
            running_train_loss = 0.0
            for imgs_b, mask_b in train_loader:
                mask_b = mask_b.squeeze(1)
                imgs_b, mask_b = imgs_b.to(device), mask_b.to(device)
                optimizer.zero_grad()
                output = model(imgs_b)
                loss = criterion(output, mask_b)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
            train_loss = running_train_loss / len(train_loader)
            logger.info(f"Epoch {ep + 1} - Train loss: {train_loss:.4f}")

            if len(train_config['trainval_split']) > 1:
                model.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for imgs_b, mask_b in val_loader:
                        mask_b = rearrange(mask_b, 'b c h w -> b (c h) w')
                        imgs_b, mask_b = imgs_b.to(device), mask_b.to(device)
                        output = model(imgs_b)
                        loss = criterion(output, mask_b)
                        running_val_loss += loss.item()
                val_loss = running_val_loss / len(val_loader)
                logger.info(f"Epoch {ep + 1} - Validation loss: {val_loss:.4f}")
        
            save_checkpoint(model, train_loss, val_loss, args.model_dir)

        logger.info("Training completed")

    elif mode == "inference":
        infer_config = {
            'test_data': os.environ.get('SM_CHANNEL_TEST_DATA', '/opt/ml/input/data')
        }
        test_data = [img_tfs(Image.open(f'{infer_config["test_data"]}/{file}')) for file in os.listdir(infer_config["test_data"])]
        test_data = torch.stack(test_data, dim=0)
        logger.info(f"Test data shape: {test_data.shape}")
        model.eval()
        with torch.no_grad():
            output = model(test_data)
        logger.info(f"Inference completed: {output}")
