import argparse
import logging
import os
import medpy.metric as metric
import numpy as np
import torch
import numpy
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from Model import *
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split


def predict_img(net,
                device,
                scale_factor=1,
                out_threshold=0.1,
                output=False):
    net.eval()
    testdata = BasicDataset(test_img, test_mask, scale_factor)
    npy_loader = DataLoader(testdata, batch_size=1, shuffle=False, num_workers=6, pin_memory=True)
    tot = len(npy_loader)
    slices_metric = []
    metric_list = []

    with tqdm(total=tot, desc='predicting', unit='npy') as pbar:

        for batch in npy_loader:
            x = batch['image']
            gt = batch['mask']
            x = x.to(device=device, dtype=torch.float32)
            gt = gt.to(device=device, dtype=torch.float32)
            with torch.no_grad():
                pred = net(x)
            # pred = torch.sigmoid(pred)
            pred = torch.squeeze(pred, dim=1).cpu().detach().numpy()
            gt = torch.squeeze(gt, dim=1).cpu().detach().numpy()
            metrics = calculate_metric_percase(pred, gt, seg_threshold=out_threshold)
            if output:
                pred[pred >= out_threshold] = 1
                pred[pred < out_threshold] = 0
                id_ = batch['id'][0] + '.png'
                tmp_dir = os.path.join(output_dir, id_)
                # slices_metric.append(cal_slice_metrics(batch['id'][0], metrics))

                pred = pred.squeeze()
                img = mask_to_image(pred)
                img.save(tmp_dir)
            metric_list.append(metrics)
            pbar.update(x.shape[0])

        avg_metricse = (np.array(metric_list)).mean(0)
    return avg_metricse


def calculate_metric_percase(pred, gt, seg_threshold=0.5):
    pred[pred >= seg_threshold] = 1
    pred[pred < seg_threshold] = 0
    gt[gt > 0] = 1
    if pred.sum() == 0 and gt.sum() == 0:
        return [1, 1, 0, 1, 1]
    elif pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        f2 = F2(pred, gt)
        hd95 = hd(pred, gt)
        pr = metric.precision(pred, gt)
        re = metric.recall(pred, gt)
        return [dice, f2, hd95, pr, re]
    else:
        return [0, 0, 373.13, 0, 0]  # nnUNet HD95 = 373.13


def F2(result, reference):
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))

    tn = numpy.count_nonzero(~result & ~reference)
    fp = numpy.count_nonzero(result & ~reference)
    fn = numpy.count_nonzero(~result & reference)
    tp = numpy.count_nonzero(result & reference)

    try:
        f2_score = 5 * tp / float(5 * tp + 4 * fn + fp)
    except ZeroDivisionError:
        f2_score = 0.0

    return f2_score


def hd(result, reference):

    if 0 == numpy.count_nonzero(reference) and 0 == numpy.count_nonzero(result):
        return 0
    if (0 != numpy.count_nonzero(reference) and 0 == numpy.count_nonzero(result)) or \
            (0 == numpy.count_nonzero(reference) and 0 != numpy.count_nonzero(result)):
        return 373.13

    return metric.hd95(result, reference)


def cal_slice_metrics(slice_id, metrics_list):

    one_slice_id_metric = [slice_id]
    one_slice_id_metric.extend(metrics_list)
    return one_slice_id_metric


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":

    net = FCN8s()
    result = []
    models_dir = 'checkpoints/FCN8s.pth'
    test_img = r'E:\U-Net_npy\dataset\test\x'
    test_mask = r'E:\U-Net_npy\dataset\test\y'
    output_dir = 'predicted_mask'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(models_dir, map_location=device))
    print("Model loaded !")
    avg_metric = predict_img(net=net, device=device)
    print(f'dice:{avg_metric[0]} \n f2:{avg_metric[1]} \n hd:{avg_metric[2]} \n pr:{avg_metric[3]} \n re:{avg_metric[4]}')
