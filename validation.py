import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from torch.utils.data import DataLoader
from PIL import Image
import network as fn

# --- CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# Chemins (Identiques à votre main.py)
base_dir = './dataset'
val_img_dir = os.path.join(base_dir, 'test')          # Les 449 images publiques
val_mask_dir = os.path.join(base_dir, 'gt', 'test_gt') # Les masques correspondants
model_path = './model/unet_cbam_voc/best_model.pt'     # Chemin vers votre fichier .pt

num_classes = 21
image_size = 256

# --- FONCTIONS UTILITAIRES ---
def get_val_lists(img_dir, mask_dir):
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
    return images, masks

def load_data_val(img_path, mask_path, size=256):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = Image.open(mask_path)
    mask = np.array(mask)
    
    # Resize strict à 256x256 comme demandé dans le PDF
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    
    # Normalisation
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    
    return torch.from_numpy(img.copy()), torch.from_numpy(mask.copy()).long()

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

def compute_pixel_acc(pred, target):
    # On ignore le label 255 (bordures)
    valid_mask = target != 255
    correct = (pred[valid_mask] == target[valid_mask]).sum().item()
    total = valid_mask.sum().item()
    if total == 0:
        return 0
    return correct / total


# --- VALIDATION ---
def validate():
    # 1. Charger les données
    print("Chargement des données de test...")
    val_imgs, val_masks = get_val_lists(val_img_dir, val_mask_dir)
    print(f"Images trouvées : {len(val_imgs)}")

    # 2. Charger le modèle
    # ATTENTION : Assurez-vous que c'est la bonne architecture (Small_UNet + CBAM)
    # Si vous n'avez pas réactivé CBAM dans network.py, commentez-le là-bas ou changez ici.
    print("Chargement du modèle...")
    model = fn.CBAM_UNet(n_channels=3, n_classes=num_classes).to(DEVICE)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("Poids chargés avec succès.")
    else:
        print(f"Erreur : Impossible de trouver {model_path}")
        return

    model.eval()
    miou_list = []
    
    print("Lancement de l'inférence avec TTA (Miroir)...")
    
    with torch.no_grad():
        for i in range(len(val_imgs)):
            # Chargement manuel (pas de DataLoader pour être sûr de l'ordre)
            v_inp, v_tar = load_data_val(val_imgs[i], val_masks[i], size=image_size)
            
            # Ajout dimension Batch [1, C, H, W]
            v_inp = v_inp.unsqueeze(0).to(DEVICE)
            v_tar = v_tar.unsqueeze(0).to(DEVICE)

            # --- TTA (Test Time Augmentation) ---
            # 1. Normal
            logits_normal = model(v_inp)
            probs_normal = torch.softmax(logits_normal, dim=1)

            # 2. Flip Horizontal
            v_inp_flip = torch.flip(v_inp, dims=[3])
            logits_flip = model(v_inp_flip)
            probs_flip = torch.softmax(logits_flip, dim=1)
            probs_flip_back = torch.flip(probs_flip, dims=[3]) # On remet à l'endroit

            # 3. Moyenne
            avg_probs = (probs_normal + probs_flip_back) / 2.0
            pred_mask = torch.argmax(avg_probs, dim=1)

            # Score
            score = compute_iou(pred_mask, v_tar, n_classes=num_classes)
            miou_list.append(score)
            
            if i % 50 == 0:
                print(f"Image {i}/{len(val_imgs)} traitée...")

    final_miou = np.nanmean(miou_list) * 100
    print("-" * 30)
    print(f"RESULTAT FINAL (Public Test Set)")
    print(f"mIoU avec TTA : {final_miou:.4f}%")
    print("-" * 30)

    # ... après le calcul du mIoU ...
    p_acc = compute_pixel_acc(pred_mask, v_tar)
    print(f"Pixel Acc: {p_acc * 100:.2f}%")


if __name__ == "__main__":
    validate()
