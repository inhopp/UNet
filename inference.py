import os
import torch
import torch.nn as nn
import torchvision
from data import generate_loader
from option import get_option
from model import UNET

@torch.no_grad()
def main(opt):
    dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
    ft_path = os.path.join(opt.ckpt_root, opt.data_name, opt.model_name, "best_epoch.pt")

    model = UNET(in_channels=3, out_channels=1).to(dev)
    if opt.multigpu:
        model = nn.DataParallel(model, device_ids=opt.device_ids).to(dev)
    model.load_state_dict(torch.load(ft_path))

    test_loader = generate_loader('test', opt)
    print("test set ready")

    model.eval()

    save_folder = "inference_images/"
    for idx, (data, targets) in test_loader:
        data = data.to(dev)
        preds = torch.sigmoid(model(data))
        preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{save_folder}/pred_{idx}.png")
        torchvision.utils.save_image(targets.unsqueeze(1), f"{save_folder}{idx}.png")

if __name__ == '__main__':
    opt = get_option()
    main(opt)