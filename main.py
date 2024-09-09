
import argparse
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from network import Network
from utils import AverageMeter
from dataset import CumstomDataset

import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.008, help='Initial learning rate for sgd.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')

    return parser.parse_args()


torch.manual_seed(0)
torch.cuda.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Network().to(device)

args = parse_args()

train_dataset = CumstomDataset(phase = 'train')
val_dataset = CumstomDataset(phase = 'val')
print('Whole train set size:', train_dataset.__len__())
print('Whole val set size:', val_dataset.__len__())

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = args.batch_size,
                                            num_workers = 4,
                                            shuffle = True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size = args.batch_size,
                                            num_workers = 4,
                                            shuffle = False)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay = 1e-4, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
best_mAP = {
        "1_to_2": 0.0,
        "2_to_1": 0.0,
    }

def train():
    for epoch in tqdm(range(1, args.epochs + 1)):
        epoch_loss = AverageMeter()
        model.train()

        for i, (t_imgs, v_imgs, targets, _) in tqdm(enumerate(train_loader), leave=False):
            optimizer.zero_grad()

            output = model(t_imgs.to(device), v_imgs.to(device), targets.to(device))
            # loss = criterion(output['y1'], targets.to(device))
            loss = output['loss']
            epoch_loss.update(loss.item(), args.batch_size)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                message = f'Train Epoch: {epoch}, loss: {epoch_loss.avg:.3f}'
                tqdm.write(message)

        eval_epoch(model)


def eval_epoch(model):
    model.eval()
    latent, obj = [[],[]], []
 

    with torch.no_grad():
        for i, (t_imgs, v_imgs, targets, obj_ids) in tqdm(enumerate(val_loader), leave=False):
            output = model(t_imgs.to(device), v_imgs.to(device), targets.to(device))
            
            latent[0].append(output['y1'].detach().cpu().numpy())
            latent[1].append(output['y2'].detach().cpu().numpy())

            obj.append([int(o) for o in obj_ids])

        
    latent[0] = np.concatenate(latent[0],axis=0) # (N, dim)
    latent[1] = np.concatenate(latent[1],axis=0)
    obj = np.concatenate(obj, axis=0) # (N,)

    mAP_1_to_2 = metrics.ranking_mAP((latent[0], latent[1]), obj)*100.
    mAP_2_to_1 = metrics.ranking_mAP((latent[1], latent[0]), obj)*100.

    if mAP_1_to_2 + mAP_2_to_1 > best_mAP['1_to_2'] + best_mAP['2_to_1']:
        best_mAP['1_to_2'] = mAP_1_to_2
        best_mAP['2_to_1'] = mAP_2_to_1

    print("mAP ({}->{}) = {:.4f} (best: {:.4f})".format('touch','vision',mAP_1_to_2, best_mAP['1_to_2']))
    print("mAP ({}->{}) = {:.4f} (best: {:.4f})".format('vision','touch',mAP_2_to_1, best_mAP['2_to_1']))
    


if __name__ == "__main__":        
    train()