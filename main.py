import os
import datetime
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk
from torch.optim.lr_scheduler import CosineAnnealingLR

# ======================= 配置 =======================
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_PATH= r"E:\my projects\finalwork\subset7"
BATCH_SIZE  = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS  = 100
IMG_SIZE    = 512
SAVE_DIR    = './saved_models'
os.makedirs(SAVE_DIR, exist_ok=True)

# ======================= 数据集切分 =======================
def create_dataset_splits(dataset_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    all_bases = [os.path.splitext(f)[0]
                 for f in os.listdir(dataset_path)
                 if f.endswith('.raw') and os.path.exists(os.path.join(dataset_path, f.replace('.raw', '.mhd')))]
    random.shuffle(all_bases)
    total = len(all_bases)
    t = int(train_ratio * total)
    v = int(val_ratio * total)
    splits = {
        'train': all_bases[:t],
        'val':   all_bases[t:t+v],
        'test':  all_bases[t+v:]
    }
    for split, bases in splits.items():
        d = os.path.join(SAVE_DIR, split)
        os.makedirs(d, exist_ok=True)
        for b in bases:
            for ext in ['.raw', '.mhd']:
                shutil.copy(os.path.join(dataset_path, b+ext), d)
    return splits['train'], splits['val'], splits['test']

# ======================= 数据增强 =======================
class GaussianNoise:
    def __init__(self, std=0.03):
        self.std = std
    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

img_transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    GaussianNoise(std=0.02)
])
mask_transform = transforms.Compose([
    transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1),
                            interpolation=transforms.InterpolationMode.NEAREST),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15)
])

# ======================= 数据集类 =======================
class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.data_dir = data_dir
        self.augment = augment
        self.bases = [os.path.splitext(f)[0]
                      for f in os.listdir(data_dir)
                      if f.endswith('.raw')]

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        b = self.bases[idx]

        # 用 SimpleITK 读取原始图像
        sitk_img = sitk.ReadImage(os.path.join(self.data_dir, b+'.mhd'))
        img_arr  = sitk.GetArrayFromImage(sitk_img)[0]  # 取第 0 层
        img = cv2.resize(img_arr, (IMG_SIZE,IMG_SIZE)).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # 读取掩码
        mask_arr = sitk.GetArrayFromImage(sitk_img)  # 和图像同大小
        mask = mask_arr[0] if mask_arr.ndim>2 else mask_arr
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask>0).astype(np.float32)

        img_t  = torch.from_numpy(img).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.augment:
            seed = torch.randint(0, 1_000_000, (1,)).item()
            torch.manual_seed(seed)
            img_t = img_transform(img_t)
            torch.manual_seed(seed)
            mask_t = mask_transform(mask_t)

        return img_t, mask_t

# 创建 splits 并加载
create_dataset_splits(DATASET_PATH)
train_set = MedicalImageDataset(os.path.join(SAVE_DIR,'train'), augment=True)
val_set   = MedicalImageDataset(os.path.join(SAVE_DIR,'val'),   augment=False)
test_set  = MedicalImageDataset(os.path.join(SAVE_DIR,'test'),  augment=False)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

# ======================= 模型定义 =======================
class DeepLabv3Plus(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
        self.early = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.mid   = backbone.layer2
        self.deep  = backbone.layer3
        self.aspp  = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256,64,1,bias=False), nn.BatchNorm2d(64), nn.ReLU()),
            nn.Sequential(nn.Conv2d(256,64,3,padding=6,dilation=6,bias=False), nn.BatchNorm2d(64), nn.ReLU())
        ])
        self.project       = nn.Sequential(nn.Conv2d(128,128,1,bias=False), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout(0.5))
        self.low_proj      = nn.Sequential(nn.Conv2d(64,24,1,bias=False), nn.BatchNorm2d(24), nn.ReLU())
        self.middle_proj   = nn.Sequential(nn.Conv2d(128,128,1,bias=False), nn.BatchNorm2d(128), nn.ReLU())
        self.classifier    = nn.Sequential(
            nn.Conv2d(128+128+24,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,num_classes,1)
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self,x):
        e = self.early(x)
        m = self.mid(e)
        d = self.deep(m)
        aspp_feats = torch.cat([conv(d) for conv in self.aspp], dim=1)
        p = self.project(aspp_feats)
        m_p = self.middle_proj(m)
        p2 = self.up2(p)
        comb = torch.cat([p2, m_p], dim=1)
        comb2 = self.up2(comb)
        low = self.low_proj(e)
        comb3 = torch.cat([comb2, low], dim=1)
        out = self.classifier(comb3)
        return self.up4(out)

# ======================= 损失与优化 =======================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice(self, p, t, eps=1e-6):
        p = torch.sigmoid(p)
        inter = (p * t).sum((1,2,3))
        denom = p.sum((1,2,3)) + t.sum((1,2,3))
        return 1 - (2 * inter + eps) / (denom + eps)

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target).mean()

model     = DeepLabv3Plus().to(DEVICE)
criterion = CombinedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# ======================= 工具函数 =======================
def plot_loss(train_losses, val_losses):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, '-o', label='Train Loss')
    plt.plot(val_losses,   '-s', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
    plt.legend(); plt.grid(True)
    fname = f'loss_{ts}.png'
    plt.savefig(fname, bbox_inches='tight'); plt.close()
    print('Saved', fname)

def save_model(model):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(SAVE_DIR, f'model_{ts}.pth')
    torch.save(model.state_dict(), path)
    print('Model saved to', path)

def evaluate(model, loader):
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            out = model(img)
            pr = (torch.sigmoid(out) > 0.5).float()
            inter = (pr * mask).sum((1,2,3))
            ps    = pr.sum((1,2,3))
            ms    = mask.sum((1,2,3))
            dices.extend(((2*inter)/(ps+ms+1e-6)).cpu().numpy())
            ious.extend((inter/(ps+ms-inter+1e-6)).cpu().numpy())
    print(f'Dice: {np.mean(dices):.4f}, IoU: {np.mean(ious):.4f}')
    return np.mean(dices), np.mean(ious)

def visualize(model, loader, num=3):
    model.eval()
    cnt = 0
    rows = num
    plt.figure(figsize=(15,5*rows))
    with torch.no_grad():
        for img, mask in loader:
            img = img.to(DEVICE)
            out = model(img)
            pr  = torch.sigmoid(out).cpu().numpy()
            im  = img.cpu().numpy()
            m   = mask.cpu().numpy()
            for j in range(im.shape[0]):
                if cnt >= num: break
                plt.subplot(rows, 3, cnt*3+1)
                plt.imshow(im[j,0], cmap='gray'); plt.axis('off'); plt.title('Image')
                plt.subplot(rows, 3, cnt*3+2)
                plt.imshow(m[j,0], cmap='gray'); plt.axis('off'); plt.title('Mask')
                plt.subplot(rows, 3, cnt*3+3)
                plt.imshow(pr[j,0], cmap='gray'); plt.axis('off'); plt.title('Pred')
                cnt += 1
            if cnt >= num: break
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f'results_{ts}.png'
    plt.tight_layout(); plt.savefig(fname, bbox_inches='tight'); plt.close()

    print('Saved', fname)
if __name__ == '__main__':
    print('Start Training...')
    train_losses, val_losses = [], []
    train_dices,  val_dices  = [], []
    train_ious,   val_ious   = [], []
    train_accs,   val_accs   = [], []   # ← 新增

    best_val_dice = 0.0
    for ep in range(1, NUM_EPOCHS+1):
        # ——— 训练 ———
        model.train()
        t_loss = t_correct = t_total = 0
        for im, ms in train_loader:
            im, ms = im.to(DEVICE), ms.to(DEVICE)
            optimizer.zero_grad()
            out = model(im)
            loss = criterion(out, ms)
            loss.backward(); optimizer.step()
            t_loss += loss.item()

            # 计算 Dice/IoU
            pr = (torch.sigmoid(out) > 0.5).float()
            inter = (pr * ms).sum((1,2,3))
            ps, ms_sum = pr.sum((1,2,3)), ms.sum((1,2,3))
            train_dices.append(((2*inter)/(ps+ms_sum+1e-6)).mean().item())
            train_ious.append((inter/(ps+ms_sum-inter+1e-6)).mean().item())

            # 计算像素准确率
            t_correct += (pr == ms).sum().item()
            t_total   += pr.numel()

        train_losses.append(t_loss / len(train_loader))
        train_accs.append(t_correct / t_total)   # ← 新增

        # ——— 验证 ———
        model.eval()
        v_loss = v_correct = v_total = 0
        dices_v = []; ious_v = []
        with torch.no_grad():
            for im, ms in val_loader:
                im, ms = im.to(DEVICE), ms.to(DEVICE)
                out = model(im)
                v_loss += criterion(out, ms).item()

                pr = (torch.sigmoid(out) > 0.5).float()
                inter = (pr * ms).sum((1,2,3))
                ps, ms_sum = pr.sum((1,2,3)), ms.sum((1,2,3))
                dices_v.extend(((2*inter)/(ps+ms_sum+1e-6)).cpu().numpy())
                ious_v.extend((inter/(ps+ms_sum-inter+1e-6)).cpu().numpy())

                v_correct += (pr == ms).sum().item()  # ← 新增
                v_total   += pr.numel()               # ← 新增

        val_losses.append(v_loss / len(val_loader))
        val_dices.append(np.mean(dices_v))
        val_ious.append(np.mean(ious_v))
        val_accs.append(v_correct / v_total)      # ← 新增

        scheduler.step()

        # —— 每轮打印 ——
        print(f"Epoch {ep}/{NUM_EPOCHS} | "
              f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f} | "
              f"Train Dice: {train_dices[-1]:.4f} | Val Dice: {val_dices[-1]:.4f} | "
              f"Train Acc: {train_accs[-1]:.4f} | Val Acc: {val_accs[-1]:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最优模型
        if val_dices[-1] > best_val_dice:
            best_val_dice = val_dices[-1]
            save_model(model)

    # —— 绘制 Loss/Dice/IoU 曲线 ——
    plot_loss(train_losses, val_losses)

    plt.figure(figsize=(10,5))
    plt.plot(train_dices, '-o', label='Train Dice')
    plt.plot(val_dices,   '-s', label='Val Dice')
    plt.xlabel('Epoch'); plt.ylabel('Dice'); plt.title('Dice Curve')
    plt.legend(); plt.grid(True)
    plt.savefig('dice_curve.png'); plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(train_ious, '-o', label='Train IoU')
    plt.plot(val_ious,   '-s', label='Val IoU')
    plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.title('IoU Curve')
    plt.legend(); plt.grid(True)
    plt.savefig('iou_curve.png'); plt.close()

    # —— 新增：绘制准确率曲线 ——
    plt.figure(figsize=(10,5))
    plt.plot(train_accs, '-o', label='Train Acc')
    plt.plot(val_accs,   '-s', label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Pixel Accuracy'); plt.title('Accuracy Curve')
    plt.legend(); plt.grid(True)
    plt.savefig('accuracy_curve.png'); plt.close()
    print('Saved accuracy_curve.png')

    # —— 最终评估与可视化 ——
    print('\nEvaluate on test set...')
    evaluate(model, test_loader)
    print('\nVisualize results...')
    visualize(model, test_loader)
    print('Done.')
