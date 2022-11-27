import os 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_loader
from option import get_option
from model import UNET

class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.model = UNET(in_channels=3, out_channels=1).to(self.dev)
        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, opt.data_name, "best_epoch.pt")
            self.model.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.device_ids).to(self.dev)

        print("# params:", sum(map(lambda x: x.numel(), self.model.parameters())))

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=opt.lr)

        self.train_loader = generate_loader('train', opt)
        print("train set ready")
        self.val_loader = generate_loader('val', opt)
        print("validation set ready")
        self.best_score, self.best_epoch = 0, 0

    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            self.model.train()
            loop = tqdm(self.train_loader)

            for _, (data, targets) in enumerate(loop):
                data = data.to(self.dev)
                targets = targets.float().unsqueeze(1).to(self.dev)
                preds = self.model(data)
                loss = self.loss_fn(preds, targets)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loop.set_postfix(loss = loss.item())

            val_score = self.eval(self.val_loader)
            
            if val_score >= self.best_score:
                self.best_score = val_score
                self.best_epoch = epoch + 1
                self.save()

            print("Epoch [{}/{}], Val dice_score: {:.3f}".format(epoch+1, opt.n_epoch, val_score))
            print("Best : {:.2f} @epoch {}".format(self.best_score, self.best_epoch+1))


    @torch.no_grad()
    def eval(self, data_loader):
        loader = data_loader
        self.model.eval()
        dice_score = 0

        for data, targets in loader:
            data = data.to(self.dev)
            targets = targets.to(self.dev).unsqueeze(1)
            preds = torch.sigmoid(self.model(data))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds*targets).sum()) / ((preds + targets).sum() + 1e-8)

        return dice_score

    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root, self.opt.data_name), exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, self.opt.data_name, "best_epoch.pt")
        torch.save(self.model.state_dict(), save_path)


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()

if __name__ == "__main__":
    main()