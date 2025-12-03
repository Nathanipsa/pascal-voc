import network as fn
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# --- CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Hyperparameters
brestore = True
restore_iter = 0
num_training_steps = 30000
learning_rate = 1e-3
weight_decay = 1e-2
batch_size = 8
accumulation_steps = 2
image_size = 256
num_classes = 21

# Paths
base_dir = './dataset'
train_img_dir = os.path.join(base_dir, 'train')
train_mask_dir = os.path.join(base_dir, 'gt', 'train_gt')
val_img_dir = os.path.join(base_dir, 'test')
val_mask_dir = os.path.join(base_dir, 'gt', 'test_gt')

model_save_path = './model/unet_cbam_voc'

# Validation / Save
val_interval = 500
best_miou = 0.0
patience = 20
patience_counter = 0

if brestore:
    path = f"{model_save_path}/best_model.pt"
    
    if os.path.exists(path):
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()
        
        # Mise à jour
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)



# --- UTILS & DATASET ---

def get_custom_lists(img_dir, mask_dir):
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Dossiers introuvables: {img_dir}")

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = sorted(os.listdir(img_dir))
    img_filenames = [f for f in all_files if f.lower().endswith(valid_extensions)]

    images = []
    masks = []
    for filename in img_filenames:
        basename = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, filename)
        mask_name = basename + ".png"
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            images.append(img_path)
            masks.append(mask_path)

    print(f"Found {len(images)} pairs in {img_dir}")
    return images, masks


class Augmenter:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, mask):
        if np.random.rand() < 0.5:
            image, mask = self.random_crop_resize(image, mask)
        if np.random.random() < 0.5:
            image, mask = self.horizontal_flip(image, mask)
        if np.random.random() < 0.5:
            image, mask = self.random_rotation(image, mask)
        if np.random.random() < 0.8:
            image = self.color_jitter(image)
        if np.random.random() < 0.2:
            image, mask = self.elastic_transform(image, mask)
        return image, mask

    def horizontal_flip(self, img, mask):
        return np.flip(img, axis=1).copy(), np.flip(mask, axis=1).copy()

    def random_rotation(self, img, mask, max_angle=15):
        h, w = img.shape[:2]
        angle = np.random.uniform(-max_angle, max_angle)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        return img_rot, mask_rot

    def random_crop_resize(self, img, mask, min_scale=0.5):
        h, w = img.shape[:2]
        scale = np.random.uniform(min_scale, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img_crop = img[top:top + new_h, left:left + new_w]
        mask_crop = mask[top:top + new_h, left:left + new_w]
        img_res = cv2.resize(img_crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_crop, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return img_res, mask_res

    def color_jitter(self, img):
        img = img.astype(np.float32)
        brightness = np.random.uniform(0.7, 1.3)
        img = img * brightness
        contrast = np.random.uniform(0.7, 1.3)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * contrast + mean
        noise_r = np.random.uniform(0.9, 1.1)
        noise_g = np.random.uniform(0.9, 1.1)
        noise_b = np.random.uniform(0.9, 1.1)
        img[:, :, 0] *= noise_r
        img[:, :, 1] *= noise_g
        img[:, :, 2] *= noise_b
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def elastic_transform(self, image, mask, alpha=100, sigma=10):
        # Simulation simple de déformation élastique via bruit gaussien sur les coordonnées
        # Nécessite scipy.ndimage (import à ajouter si vous l'avez, sinon sautez cette fonction)
        # Pour faire simple sans scipy, on peut faire un shift simple :
        tx = np.random.randint(-10, 10)
        ty = np.random.randint(-10, 10)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        rows, cols = image.shape[:2]
        img_shift = cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))
        mask_shift = cv2.warpAffine(mask, M, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=255)
        return img_shift, mask_shift

augmenter = Augmenter(size=256)


def load_data(img_path, mask_path, size=256, augment=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = Image.open(mask_path)
    mask = np.array(mask)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    if augment:
        img, mask = augmenter(img, mask)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return img.copy(), mask.copy()


class VOCDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=False, size=256):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.size = size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img, mask = load_data(self.img_paths[idx], self.mask_paths[idx],
                              size=self.size, augment=self.augment)
        return torch.from_numpy(img), torch.from_numpy(mask).long()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        # Target : (B, H, W) -> One hot : (B, C, H, W)
        target = self._one_hot_encoder(target)

        if inputs.size() != target.size():
            target = nn.functional.interpolate(target, size=inputs.shape[2:], mode='nearest')

        # Intersection and Union
        smooth = 1e-5
        intersection = inputs * target

        dice_part = (2. * intersection.sum(dim=(2, 3)) + smooth) / (
                    inputs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)

        # Dice Loss = 1 - Dice Score moyen
        return 1 - dice_part.mean()



def compute_iou(pred, target, n_classes=21):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    valid_mask = target != 255
    pred = pred[valid_mask]
    target = target[valid_mask]
    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)


# --- INIT ---
train_imgs, train_masks = get_custom_lists(train_img_dir, train_mask_dir)
val_imgs, val_masks = get_custom_lists(val_img_dir, val_mask_dir)

train_dataset = VOCDataset(train_imgs, train_masks, augment=True)
val_dataset = VOCDataset(val_imgs, val_masks, augment=False)

# if launched on Windows : num_workers=0; if launched on Linux : num_workers=4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# Model
model = fn.CBAM_UNet(n_channels=3, n_classes=num_classes).to(DEVICE)

# --- INIT ---

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_training_steps, eta_min=1e-6
)

# --- DEFINITION DES PERTES ---
# 1. Cross Entropy : On garde un poids léger sur le fond (0.4) pour la stabilité
class_weights = torch.ones(num_classes).to(DEVICE)
class_weights[0] = 0.4
criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

# 2. Dice Loss : Pour la précision des formes
criterion_dice = DiceLoss(n_classes=num_classes)

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

# --- Learning ---
print("Starting training...")
optimizer.zero_grad()
global_step = 0
training_active = True

# We go as long as training is active (i.e. patience counter is lower than patience and it is lower
while training_active:

    for batch_img, batch_mask in train_loader:
        global_step += 1
        model.train()

        # 1. Data
        inputs = batch_img.to(DEVICE)
        targets = batch_mask.to(DEVICE)

        # 2. Forward
        outputs = model(inputs)

        # 3. Loss & Backward
        loss_ce = criterion_ce(outputs, targets)
        loss_dice = criterion_dice(outputs, targets, softmax=True)

        # On combine : CE stabilise le début, Dice affine la fin.
        # Vous pouvez ajuster les coefficients, mais 0.5/0.5 est un standard robuste.
        total_loss = (0.5 * loss_ce + 0.5 * loss_dice)

        # Division par accumulation_steps car on est dans la boucle d'accumulation
        (total_loss / accumulation_steps).backward()

        # Pour l'affichage dans le print, on garde la valeur brute
        loss = total_loss

        # 4. Optimization
        if global_step % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # 5. Logging
        if global_step % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Iter: {global_step:05d} | Loss: {loss.item() * accumulation_steps:.4f} | LR: {current_lr:.6f}")

        # 6. Validation
        if global_step % val_interval == 0:
            print('Validation...')
            model.eval()
            miou_list = []

            with torch.no_grad():
                for v_img, v_mask in val_loader:
                    v_inp = v_img.to(DEVICE)
                    v_tar = v_mask.to(DEVICE)

                    pred_logits = model(v_inp)
                    pred_mask = torch.argmax(pred_logits, dim=1)

                    score = compute_iou(pred_mask, v_tar, n_classes=num_classes)
                    miou_list.append(score)

            avg_miou = np.nanmean(miou_list) * 100
            print(f"Validation mIoU: {avg_miou:.2f}% (Best: {best_miou:.2f}%)")

            # Save Checkpoint
            torch.save(model.state_dict(), f"{model_save_path}/model_last.pt")
            if avg_miou > best_miou:
                best_miou = avg_miou
                patience_counter = 0
                print(f"New Best Model! Saving...")
                torch.save(model.state_dict(), f"{model_save_path}/best_model.pt")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early Stopping triggered.")
                training_active = False
                break

        # Stop condition
        if global_step >= num_training_steps:
            print("Max steps reached.")
            training_active = False
            break

print(f"Training finished. Best mIoU: {best_miou:.2f}%")
