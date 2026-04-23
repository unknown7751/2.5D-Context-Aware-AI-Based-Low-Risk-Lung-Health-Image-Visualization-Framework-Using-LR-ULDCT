import os
import glob
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ── 1. Recreate the DnCNN Architecture ──────────────────────────────
class DnCNN2D(nn.Module):
    def __init__(self, depth=17, num_features=64):
        super().__init__()
        layers = [nn.Conv2d(1, num_features, 3, padding=1, bias=False), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(num_features, num_features, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True)
            ])
        layers.append(nn.Conv2d(num_features, 1, 3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)

# ── 2. Configuration ────────────────────────────────────────────────
RAW_DATA_DIR = "./Model_Ready_Data/COVID-S1"                # Normal patient data
OUT_DATA_DIR = "./Model_Ready_Data/Denoised_COVID"          # Denoised output folder
WEIGHTS_PATH = "./checkpoints/dncnn2d_epoch_0100.pth"        # Your saved weights

# ── Split: First 32 for denoising, remaining 24 reserved for SRGAN test set ──
DENOISE_COUNT = 104  # N001 - N032 → denoise
# N033 - N056 (24 files) → reserved for SRGAN test set (untouched)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DATA_DIR, exist_ok=True)

# ── 3. Execution ────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading weights from {WEIGHTS_PATH}...")
    model = DnCNN2D().to(device)
    
    # Load the state dict properly (handling the dictionary format you saved)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # CRITICAL: Put model in inference mode
    
    # Get all files and sort them to ensure correct order (N001, N002, ...)
    patient_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.npy")))
    print(f"Found {len(patient_files)} total Normal patient volumes.")
    
    # Split: first 32 for denoising, remaining 24 for SRGAN test set
    denoise_files = patient_files[:DENOISE_COUNT]
    test_files    = patient_files[DENOISE_COUNT:]
    
    print(f"  → Denoising:  {len(denoise_files)} volumes (N001-N032)")
    print(f"  → SRGAN Test:  {len(test_files)} volumes reserved (N033-N056)")
    print(f"  → Device: {device}\n")

    with torch.no_grad():
        for file_path in tqdm(denoise_files, desc="Denoising Normal Patients"):
            filename = os.path.basename(file_path)
            
            # Load the raw 3D volume [Depth, 512, 512]
            raw_vol = np.load(file_path).astype(np.float32)
            clean_vol = np.zeros_like(raw_vol)
            
            # Process slice by slice (Fast enough on a 4090)
            for i in range(raw_vol.shape[0]):
                slice_2d = raw_vol[i]
                
                # Convert to tensor [1, 1, 512, 512]
                tensor_slice = torch.from_numpy(slice_2d).unsqueeze(0).unsqueeze(0).to(device)
                
                # Use bfloat16 for speed
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    denoised_tensor = model(tensor_slice)
                
                # Squeeze back to [512, 512] and save to our clean volume
                clean_vol[i] = denoised_tensor.squeeze().cpu().numpy()
            
            # Save the fully denoised 3D volume to the new folder
            np.save(os.path.join(OUT_DATA_DIR, filename), clean_vol)

    print("\n✅ Denoising Complete!")
    print(f"  → {len(denoise_files)} denoised volumes saved to: {OUT_DATA_DIR}")
    print(f"  → {len(test_files)} raw volumes remain in {RAW_DATA_DIR} for SRGAN test set")