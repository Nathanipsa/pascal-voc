import network as fn
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from PIL import Image

# --- CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


num_training_steps = 30000
learning_rate = 1e-3  # Works for AdamW
weight_decay = 1e-2  # Same
batch_size = 8  # Takes RAM but can be higher
accumulation_steps = 2  # Works as a batch size of 16
image_size = 256
num_classes = 21  # 20 classes + background

# path
base_dir = './dataset' # Dossier racine

train_img_dir = os.path.join(base_dir, 'train')
train_mask_dir = os.path.join(base_dir, 'gt', 'train_gt')

val_img_dir = os.path.join(base_dir, 'test')
val_mask_dir = os.path.join(base_dir, 'gt', 'test_gt')

model_save_path = './model/unet_cbam_voc'


# Restauration
brestore = False
restore_iter = 0

# Validation and save best model
val_interval = 500
best_miou = 0.0
patience = 20
patience_counter = 0


# --- UTILS FOR DATASET PASCAL VOC ---
def get_custom_lists(img_dir, mask_dir):
    """
    Récupère les chemins des images et des masques en scannant les dossiers.
    Associe les fichiers par leur nom (sans extension).
    """
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Les dossiers {img_dir} ou {mask_dir} n'existent pas.")

    # Liste tous les fichiers images (filtrer par extensions si besoin)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = sorted(os.listdir(img_dir))
    img_filenames = [f for f in all_files if f.lower().endswith(valid_extensions)]

    images = []
    masks = []

    for filename in img_filenames:
        # Récupère le nom sans extension (ex: "image_01.jpg" -> "image_01")
        basename = os.path.splitext(filename)[0]
        
        # Construit le chemin de l'image
        img_path = os.path.join(img_dir, filename)
        
        # Cherche le masque correspondant (on suppose qu'il est en .png)
        mask_name = basename + ".png"
        mask_path = os.path.join(mask_dir, mask_name)

        if os.path.exists(mask_path):
            images.append(img_path)
            masks.append(mask_path)
        else:
            # Optionnel : avertir si un masque manque
            # print(f"Warning: Masque non trouvé pour {filename}")
            pass

    print(f"Trouvé {len(images)} paires images/masques dans {img_dir}")
    return images, masks

class Augmenter:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, mask):

        if np.random.rand() < 0.5:
            image, mask = self.random_crop_resize(image, mask)

        # 2. Random Horizontal Flip
        if np.random.random() < 0.5:
            image, mask = self.horizontal_flip(image, mask)

        # 3. Random Rotation
        if np.random.random() < 0.5:
            image, mask = self.random_rotation(image, mask)

        # 4. Color Jitter
        if np.random.random() < 0.8:
            image = self.color_jitter(image)

        # 5. Gaussian Noise
        if np.random.random() < 0.2:
            image = self.add_noise(image)

        return image, mask

    def horizontal_flip(self, img, mask):
        # Flip numpy
        return np.flip(img, axis=1).copy(), np.flip(mask, axis=1).copy()

    def random_rotation(self, img, mask, max_angle=15):
        h, w = img.shape[:2]
        angle = np.random.uniform(-max_angle, max_angle)

        # Calculation of rotation matrix
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)

        return img_rot, mask_rot

    def random_crop_resize(self, img, mask, min_scale=0.5):
        h, w = img.shape[:2]

        # New size of the image
        scale = np.random.uniform(min_scale, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)

        # Choosing the corner
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # Crop
        img_crop = img[top:top + new_h, left:left + new_w]
        mask_crop = mask[top:top + new_h, left:left + new_w]

        # Resize back to original size (256x256)
        img_res = cv2.resize(img_crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask_res = cv2.resize(mask_crop, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        return img_res, mask_res

    def color_jitter(self, img):

        img = img.astype(np.float32)

        # Brightness
        brightness = np.random.uniform(0.7, 1.3)
        img = img * brightness

        # Contraste
        contrast = np.random.uniform(0.7, 1.3)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        img = (img - mean) * contrast + mean

        # Saturation (via conversion HSV)
        noise_r = np.random.uniform(0.9, 1.1)
        noise_g = np.random.uniform(0.9, 1.1)
        noise_b = np.random.uniform(0.9, 1.1)
        img[:, :, 0] *= noise_r
        img[:, :, 1] *= noise_g
        img[:, :, 2] *= noise_b

        # Clip pour rester entre 0 et 255
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def add_noise(self, img):
        # Gaussian Noise
        mean = 0
        sigma = np.random.uniform(5, 20)  # Noise intensity
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) + gauss
        return np.clip(noisy, 0, 255).astype(np.uint8)

augmenter = Augmenter(size=256)

def load_data(img_path, mask_path, size=256, augment=False):
    """Loads and preprocesses an image and its corresponding mask."""
    # Image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Mask
    mask = Image.open(mask_path)
    mask = np.array(mask)

    # Resize
    img  = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    if augment:
        img, mask = augmenter(img, mask)

    # Normalisation Image
    img = img.astype(np.float32) / 255.0
    # Mean/Std ImageNet (standard)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) # Random values => need to change when we have the dataset
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) # Same
    img = (img - mean) / std

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    return img.copy(), mask.copy()


def get_batch(image_list, mask_list, batch_size, augment=True):
    """Générateur de batch simple"""
    indices = np.random.choice(len(image_list), batch_size)
    batch_x, batch_y = [], []

    for idx in indices:
        img, mask = load_data(image_list[idx], mask_list[idx], size=image_size, augment=augment)
        batch_x.append(img)
        batch_y.append(mask)

    return np.array(batch_x), np.array(batch_y)


def compute_iou(pred, target, n_classes=21):
    """Calcul du mIoU"""
    # pred: [B, H, W] (indices), target: [B, H, W]
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore background (255) in VOC
    valid_mask = target != 255
    pred = pred[valid_mask]
    target = target[valid_mask]

    for cls in range(n_classes):
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))  # Ignorer cette classe si pas présente
        else:
            ious.append(intersection / union)

    # Moyenne en ignorant les NaNs
    return np.nanmean(ious)


# --- PREPARATION DATA ---

print('Loading Dataset Paths...')

train_imgs, train_masks = get_custom_lists(train_img_dir, train_mask_dir)
val_imgs, val_masks = get_custom_lists(val_img_dir, val_mask_dir)

# Vérification de sécurité
if len(train_imgs) == 0:
    raise RuntimeError("Aucune image d'entraînement trouvée. Vérifiez vos chemins.")
if len(val_imgs) == 0:
    raise RuntimeError("Aucune image de validation trouvée. Vérifiez vos chemins.")
# --- MODELE & OPTIM ---

#model = fn.UNet(n_channels=3, n_classes=num_classes).to(DEVICE)

model = fn.CBAM_UNet(n_channels=3, n_classes=num_classes).to(DEVICE)

# Optimiseur AdamW (Mieux que SGD pour training from scratch)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Scheduler Cosine
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps, eta_min=1e-6)

# Loss : CrossEntropy ignore_index=255 (border)
criterion = nn.CrossEntropyLoss(ignore_index=255)

# Restauration
if brestore:
    checkpoint_path = f"{model_save_path}/model_{restore_iter}.pt"
    if os.path.exists(checkpoint_path):
        print(f'Restoring model from {checkpoint_path}')
        model.load_state_dict(torch.load(checkpoint_path))
        # Optionnel : restaurer scheduler
        for _ in range(restore_iter): scheduler.step()
    else:
        print("Checkpoint not found, starting from scratch.")

if not os.path.isdir(model_save_path):
    os.makedirs(model_save_path)

# --- BOUCLE D'ENTRAINEMENT ---

print("Starting training...")
optimizer.zero_grad()

for it in range(restore_iter, num_training_steps + 1):
    model.train()

    # Get Batch
    batch_img, batch_mask = get_batch(train_imgs, train_masks, batch_size, augment=True)

    inputs = torch.from_numpy(batch_img).to(DEVICE)
    targets = torch.from_numpy(batch_mask).long().to(DEVICE)

    # Forward
    outputs = model(inputs)

    # Loss
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    # Update Weights
    if (it + 1) % accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Logging
    if it % 100 == 0:
        print(
            f"Iter: {it:05d} | Loss: {loss.item() * accumulation_steps:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # --- VALIDATION ---
    if it % val_interval == 0 and it != 0:
        print('Validation...')
        model.eval()

        miou_list = []

        # On valide sur tout le set de validation (par batch de 1 pour simplifier le code)
        with torch.no_grad():
            for i in range(len(val_imgs)):
                v_img, v_mask = load_data(val_imgs[i], val_masks[i], size=image_size, augment=False)

                v_inp = torch.from_numpy(v_img).unsqueeze(0).to(DEVICE)
                v_tar = torch.from_numpy(v_mask).unsqueeze(0).long().to(DEVICE)

                pred_logits = model(v_inp)
                pred_mask = torch.argmax(pred_logits, dim=1)  # [1, H, W]

                # Calcul mIoU pour cette image
                score = compute_iou(pred_mask, v_tar, n_classes=num_classes)
                miou_list.append(score)

        avg_miou = np.nanmean(miou_list) * 100
        print(f"Validation mIoU: {avg_miou:.2f}% (Best: {best_miou:.2f}%)")

        # Save Log
        with open(model_save_path + '/log.txt', 'a+') as f:
            f.write(f"Iter: {it} | mIoU: {avg_miou:.4f}\n")

        # Checkpoint Regular
        torch.save(model.state_dict(), f"{model_save_path}/model_last.pt")

        # Save Best
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
            break

print(f"Training finished. Best mIoU: {best_miou:.2f}%")
