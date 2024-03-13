import argparse
import logging
import os
import sys
from torch import optim
from tqdm import tqdm
from eval import eval_net
from Model import *
from dice_loss import *

from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'


def train_net(net,
              device,
              epochs=20,
              batch_size=8,
              lr=0.001,
              val_percent=0.2,
              save_cp=True,
              img_scale=1):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20)

    criterion = DiceLoss()
    iter_num = 0
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == 1, \
                    f'Network has been defined with 1 input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                # imgs = imgs.permute(0, 3, 1, 2)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                iter_num = iter_num + 1
                pbar.update(imgs.shape[0])
                global_step += 1
            pbar.close()
            logging.info(f'Epoch{epoch + 1}_loss: {epoch_loss * batch_size / n_train}')
            # ------------------------------------------------------------------------------------ #
            val_dice = eval_net(net, val_loader, device)
            # scheduler.step(val_score)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iter_num)
            logging.info('Validation DSC: {}'.format(val_dice))
            writer.add_scalar('Dice/test', val_dice, iter_num)
        # scheduler.step()
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + 'agunet_model.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():

    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks.')

    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=25,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=8,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net = FRPNet()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
