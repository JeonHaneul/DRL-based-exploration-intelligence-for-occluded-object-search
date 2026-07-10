import os
import random
import argparse
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, optim
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from torchvision.transforms import Normalize
from tqdm import tqdm


class Config:
    X_DATA_DIR = "./train_x"  # x 데이터 경로
    Y_DATA_DIR = "./train_y"  # y 데이터 경로
    OUTPUT_DIR = "./outputs"  # 모델 저장 경로
    BATCH_SIZE = 16  # 배치 크기
    LR = 1e-4
    EPOCHS = 100
    CLASS_NAMES = ["can_1", "can_2", "can_3", "can_4", "cup_1", "cup_2", "cup_3", "cup_4", "mug_1", "mug_2", "mug_3", "mug_4", "bottle_1", "bottle_2", "bottle_3", "bottle_4"]  # 클래스 이름 배열
    NUM_CLASSES = len(CLASS_NAMES)
    LEARNING_DATA_RATIO = 0.8  # 학습 데이터 비율 (0.8 = 80%)
    MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_45.pth")  # 기존 모델 경로
    SAVE_IMAGES_INTERVAL = 2  # 이미지 저장 간격 (에폭 단위)


class FCNDataset(Dataset):
    def __init__(self, data_paths, transforms=None):
        assert len(data_paths) > 0, "Data paths list is empty!"

        # data_paths를 이미지와 레이블 경로로 분리
        self.image_paths = [path[0] for path in data_paths]
        self.label_paths = [path[1] for path in data_paths]

        # 클래스 이름에서 인덱스를 추출
        self.class_indices = [Config.CLASS_NAMES.index(os.path.basename(os.path.dirname(path[0]))) for path in data_paths]

        self.transforms = transforms

        # Find the minimum width and height across the dataset
        self.target_size = self.find_min_size()

        print(f"Resizing all images to the smallest dimensions: {self.target_size}")

    def find_min_size(self):
        min_width = float('inf')
        min_height = float('inf')

        for img_path in self.image_paths:
            with Image.open(img_path) as img:
                width, height = img.size
                min_width = min(min_width, width)
                min_height = min(min_height, height)

        return (min_height, min_width)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        lbl = Image.open(self.label_paths[index])

        # Process X data (Grayscale to RGB if needed)
        img = np.array(img)
        if img.ndim == 2:  # Grayscale image
            img = np.stack([img] * 3, axis=-1)  # HxW → HxWx3
        elif img.shape[-1] == 4:  # RGBA image
            img = img[..., :3]  # Drop alpha channel

        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = img.transpose(2, 0, 1)  # HxWxC → CxHxW

        # Apply Normalize transform
        if self.transforms:
            img = torch.tensor(img, dtype=torch.float32)
            img = self.transforms(img)
        else:
            img = torch.tensor(img, dtype=torch.float32)

        # Process Y data
        lbl = np.array(lbl)
        lbl = lbl.astype(np.float32) / 255.0  # Normalize to [0, 1]
        lbl = torch.tensor(lbl, dtype=torch.float32)

        # Return class index along with img and lbl
        class_idx = self.class_indices[index]
        return img, lbl, class_idx


def load_and_split_data(x_dir, y_dir, learning_data_ratio):
    def load_data(dir_path, class_names):
        data_paths = []
        for class_name in class_names:
            class_dir = os.path.join(dir_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data_paths.append(file_path)
        return sorted(data_paths)

    # Load X and Y paths
    x_paths = load_data(x_dir, Config.CLASS_NAMES)
    y_paths = load_data(y_dir, Config.CLASS_NAMES)

    assert len(x_paths) == len(y_paths), "Mismatch in number of images and labels!"

    # Combine X and Y paths
    combined_paths = list(zip(x_paths, y_paths))
    random.shuffle(combined_paths)

    # Split into training and validation sets
    split_idx = int(len(combined_paths) * learning_data_ratio)
    train_paths = combined_paths[:split_idx]
    val_paths = combined_paths[split_idx:]

    return train_paths, val_paths


class FCNModel(nn.Module):
    def __init__(self):
        super(FCNModel, self).__init__()
        self.model = fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, Config.NUM_CLASSES, kernel_size=1)  # Adjust output classes

    def forward(self, x):
        return self.model(x)['out']


def save_images(epoch, batch_idx, imgs, lbls, preds, output_dir, mean, std, class_indices, model=None):
    """
    Save input images, labels, and predictions for each class.

    Args:
        epoch: Current epoch number.
        imgs: Batch of input images.
        lbls: Batch of labels.
        preds: Batch of predictions.
        output_dir: Base output directory for saving images.
        mean: Mean for normalization.
        std: Standard deviation for normalization.
        class_indices: List of class indices for each image in the batch.
    """
    
    # if batch_idx > 0:  # Only save images for the first batch
    #     return
    
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    for idx in range(len(imgs)):
        # Get the class name for the current image
        class_name = Config.CLASS_NAMES[class_indices[idx]]

        # Create a directory for the current class
        class_dir = os.path.join(epoch_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Reverse normalization for input image
        img = imgs[idx].cpu().numpy().transpose(1, 2, 0)  # CxHxW -> HxWxC
        img = (img * std + mean) * 255  # Reverse normalization
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Process label
        lbl = lbls[idx].cpu().numpy()
        lbl = (lbl * 255).astype(np.uint8)

        # Process prediction
        pred = preds[idx].cpu().numpy()
        pred = (pred * 255)
        pred = np.clip(pred, 0, 255).astype(np.uint8)

        # Save the input, label, and prediction images
        Image.fromarray(img).save(os.path.join(class_dir, f"input_{idx}.png"))
        Image.fromarray(lbl).save(os.path.join(class_dir, f"label_{idx}.png"))
        Image.fromarray(pred).save(os.path.join(class_dir, f"prediction_{idx}.png"))
        
    if model:
        model_path = os.path.join(epoch_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_path)
        # print(f"Saved model for epoch {epoch} at {model_path}")


def train_one_epoch(model, data_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    for batch_idx, (imgs, lbls, class_indices) in enumerate(tqdm(data_loader)):
        imgs, lbls, class_indices = imgs.to(device), lbls.to(device), class_indices.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(imgs)

        # Select specific class channel based on class index
        batch_size = lbls.size(0)
        outputs = outputs[range(batch_size), class_indices]  # Select class-specific channels

        # Compute loss
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, data_loader, criterion, device, epoch, output_dir, mean, std):
    model.eval()
    total_loss = 0
    total_metrics = []

    with torch.no_grad():
        for batch_idx, (imgs, lbls, class_indices) in enumerate(tqdm(data_loader)):
            imgs, lbls, class_indices = imgs.to(device), lbls.to(device), class_indices.to(device)
            outputs = model(imgs)

            # Select specific class channel based on class index
            batch_size = lbls.size(0)
            outputs = outputs[range(batch_size), class_indices]  # Select class-specific channels

            # Compute loss
            loss = criterion(outputs, lbls)
            total_loss += loss.item()

            # Save images and model at specified intervals
            if epoch % Config.SAVE_IMAGES_INTERVAL == 0:
                save_images(epoch, batch_idx, imgs, lbls, outputs, output_dir, mean, std, class_indices.cpu().numpy(), model=model)
                
            # Convert to 0~255 for evaluation
            # lbl_pred = (outputs.cpu().numpy() * 255).astype(np.uint8)  # Predictions to 0~255
            # lbl_true = (lbls.cpu().numpy() * 255).astype(np.uint8) 
            lbl_true = lbls.cpu().numpy()
            lbl_true = (lbl_true * 255)
            lbl_true = np.clip(lbl_true, 0, 255).astype(np.float32)
            lbl_pred = outputs.cpu().numpy()
            lbl_pred = (lbl_pred * 255)
            lbl_pred = np.clip(lbl_pred, 0, 255).astype(np.float32)

            for lt, lp in zip(lbl_true, lbl_pred):
                total_metrics.append(label_accuracy_values(lt, lp))

    # Aggregate metrics
    metrics_sum = np.sum(np.array(total_metrics), axis=0)
    mean_acc, balanced_acc, mean_iou = label_accuracy_scores(*metrics_sum)

    avg_loss = total_loss / len(data_loader)
    # print(f"Validation Loss: {avg_loss:.4f}")
    # print(f"Mean Accuracy: {mean_acc:.4f}, Balanced Accuracy: {balanced_acc:.4f}, Mean IoU: {mean_iou:.4f}")

    return avg_loss, mean_acc, balanced_acc, mean_iou

def label_accuracy_values(label_trues, label_preds, pos_thresh=25.5, diff_thresh=51):
    """Calculate true positive, true negative, and related metrics for accuracy evaluation."""
    lp = np.array(label_preds)
    lt = np.array(label_trues)

    true_pos = np.sum(np.logical_and(np.abs(lp - lt) < diff_thresh, lt > pos_thresh))
    true_neg = np.sum(np.logical_and(np.abs(lp - lt) < diff_thresh, lt <= pos_thresh))
    num_pred_pos = np.sum(lt > pos_thresh)
    num_total_pos = np.sum(np.logical_or(lt > pos_thresh, lp > pos_thresh))
    return true_pos, true_neg, num_pred_pos, num_total_pos, lt.size


def label_accuracy_scores(true_pos, true_neg, num_pred_pos, num_total_pos, total_size):
    """Compute Mean Accuracy, Balanced Accuracy, and Mean IoU."""
    acc = (true_pos + true_neg) / total_size
    bal_acc = 0.5 * ((true_pos / num_pred_pos) + (true_neg / (total_size - num_pred_pos)))
    iou = true_pos / num_total_pos
    return acc, bal_acc, iou


def load_mean_std(stats_file):
    """Load mean and std from a stats file."""
    with open(stats_file, "r") as f:
        lines = f.readlines()
        mean = np.array(eval(lines[0].split(": ")[1].strip()))
        std = np.array(eval(lines[1].split(": ")[1].strip()))
    return mean, std


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change in validation loss to be considered as an improvement.
            verbose (bool): Whether to print a message when early stopping is triggered.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def log_to_file(epoch, train_loss, val_loss, mean_acc, bal_acc, mean_iou, log_file_path):
    """
    Logs training metrics to a file.

    Args:
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        val_loss (float): Validation loss.
        mean_acc (float): Mean accuracy.
        bal_acc (float): Balanced accuracy.
        mean_iou (float): Mean IoU.
        log_file_path (str): Path to the log file.
    """
    with open(log_file_path, "a") as log_file:  # 'a' mode for appending
        log_file.write(f"Epoch {epoch}:\n")
        log_file.write(f"Train Loss: {train_loss:.4f}\n")
        log_file.write(f"Val Loss: {val_loss:.4f}\n")
        log_file.write(f"Mean Accuracy: {mean_acc:.4f}\n")
        log_file.write(f"Balanced Accuracy: {bal_acc:.4f}\n")
        log_file.write(f"Mean IoU: {mean_iou:.4f}\n")
        log_file.write("=" * 50 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume training from the saved model")
    args = parser.parse_args()

    train_paths, val_paths = load_and_split_data(Config.X_DATA_DIR, Config.Y_DATA_DIR, Config.LEARNING_DATA_RATIO)
    
    # mean과 std를 파일에서 불러오기
    stats_file = os.path.join(Config.OUTPUT_DIR, "dataset_stats.txt")
    mean, std = load_mean_std(stats_file)
    print(f"Loaded Dataset mean: {mean}, std: {std}")

    # Normalize에 계산된 mean과 std 적용
    transforms = Normalize(mean=mean.tolist(), std=std.tolist())

    # FCNDataset을 데이터 병합 후 초기화
    train_dataset = FCNDataset(train_paths, transforms=transforms)
    val_dataset = FCNDataset(val_paths, transforms=transforms)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCNModel()

    if args.resume and os.path.exists(Config.MODEL_PATH):
        print(f"Resuming training from {Config.MODEL_PATH}")
        model.load_state_dict(torch.load(Config.MODEL_PATH))
    else:
        print("Starting training from scratch")

    model = model.to(device)

    criterion = nn.MSELoss()  # Mean Squared Error Loss for continuous masks
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    early_stopping = EarlyStopping(patience=20, min_delta=0.0005, verbose=True)  # EarlyStopping 객체 생성

    best_val_loss = float('inf')
    log_file_path = os.path.join(Config.OUTPUT_DIR, "log.txt")  # Log 파일 경로 지정
    
    for epoch in range(1, Config.EPOCHS + 1):  # Epoch starts from 1
        print(f"Epoch {epoch}/{Config.EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, mean_acc, bal_acc, mean_iou = validate(
            model, val_loader, criterion, device, epoch, Config.OUTPUT_DIR, mean, std
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean Accuracy: {mean_acc:.4f}, Balanced Accuracy: {bal_acc:.4f}, Mean IoU: {mean_iou:.4f}")
        
        # Log to file
        log_to_file(epoch, train_loss, val_loss, mean_acc, bal_acc, mean_iou, log_file_path)
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print("Saved Best Model")

if __name__ == "__main__":
    main()