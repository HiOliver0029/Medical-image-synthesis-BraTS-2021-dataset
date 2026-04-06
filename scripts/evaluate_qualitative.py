import torch
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from dlmi_hw1.data.datasets import PairedSliceDataset
from dlmi_hw1.models.cyclegan import Generator
from torch.utils.data import DataLoader

def save_samples(config_path, checkpoint_path, out_dir="docs/qualitative_results"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen_cfg = {'in_channels': cfg['model']['in_channels'], 'out_channels': cfg['model']['out_channels'], 'ngf': cfg['model']['ngf'], 'n_res_blocks': cfg['model']['n_res_blocks']}; g_ab = Generator(**gen_cfg).to(device)
    g_ab.load_state_dict(torch.load(checkpoint_path, map_location=device)['g_ab_state'])
    g_ab.eval()
    
    dataset = PairedSliceDataset(Path(cfg['data']['processed_root']), split='val')
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    real_a, real_b = next(iter(loader))
    real_a = real_a.to(device)
    with torch.no_grad():
        fake_b = g_ab(real_a).cpu()
    
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    for i in range(4):
        axes[i, 0].imshow(real_a[i, 0].cpu(), cmap='gray')
        axes[i, 0].set_title('Real T1 (Source)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(fake_b[i, 0], cmap='gray')
        axes[i, 1].set_title('Fake T2 (Generated)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(real_b[i, 0], cmap='gray')
        axes[i, 2].set_title('Real T2 (Target)')
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "sample_results.png")
    print(f"Sample results saved to {out_dir}/sample_results.png")

if __name__ == "__main__":
    save_samples("configs/cyclegan_brats.yaml", "checkpoints/cyclegan_epoch_020.pt")
