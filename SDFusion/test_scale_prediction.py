from models.networks.ply_networks.pointnet2 import PointNet4ScalePrediction
import torch
from torch import nn
from datasets.gapnet_dataset import GAPartNetDataset4ScalePrediction
import argparse
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch import optim
from utils.util import AverageMeter, Logger
import os
from tqdm import tqdm

parser = argparse.ArgumentParser("Train scale prediction network")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--gpu_ids", type=str, default="7")
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--part_category", type=str, default="slider_drawer")
parser.add_argument(
    "--dataroot",
    type=str,
    default="/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset",
)
parser.add_argument("--ply_rotate", action="store_true", help="rotate the ply files")

parser.add_argument(
    "--hidden_dim", type=int, default=256, help="hidden dim of PointNet++"
)

parser.add_argument(
    "--cond_ckpt",
    type=str,
    default="/raid/haoran/Project/PartDiffusion/PartDiffusion/pretrained_checkpoint/pointnet2.pth",
    help="condition model ckpt to load.",
)
parser.add_argument('--loss_fn', type=str, default='l2', help='loss function', choices=['l1', 'l2'])

args = parser.parse_args()

def main():

    experiment_path = os.path.join('./logs_scale_pred', args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    logger = Logger(os.path.join(experiment_path, 'test_log.txt'))

    model = PointNet4ScalePrediction()

    pretrained_weight = torch.load(os.path.join(experiment_path, 'checkpoint/model_latest.pth'))
    model.load_state_dict(pretrained_weight['model_state_dict'])

    train_dataset, test_dataset = GAPartNetDataset4ScalePrediction(
        args, phase="train", cat=args.part_category
    ), GAPartNetDataset4ScalePrediction(args, phase="test", cat=args.part_category)

    train_dataloader, test_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    ), DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    accelerator = Accelerator()
    device = accelerator.device

    model, train_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, test_dataloader)

    loss_fn = nn.functional.l1_loss if args.loss_fn == 'l1' else nn.functional.mse_loss

    train_loss_meter, test_loss_meter = AverageMeter(), AverageMeter()
    train_loss_meter.reset()
    test_loss_meter.reset()

    # gather the ratio between gt and estimation of individual instances, falls between [0.75, 1.25], bucket size = 0.0125
    bucket_size = 0.0125
    bucket_min, bucket_max = 0.75, 1.25
    num_bucket = int((bucket_max - bucket_min) / bucket_size)
    train_bucket, test_bucket = [0] * num_bucket, [0] * num_bucket

    # evaluation
    model.eval()

    with torch.no_grad():
        for data in tqdm(train_dataloader):
            
            ply_data = data['ply'].to(device)
            ply_scale = data['ply_scale'].to(device)
            gt_extent = data['part_extent'].to(device)

            estimated_extent = model(ply_data) * ply_scale

            loss = loss_fn(estimated_extent, gt_extent)

            estimated_extent_gathered = accelerator.gather(estimated_extent)
            gt_extent_gathered = accelerator.gather(gt_extent)
            ratio = estimated_extent_gathered / gt_extent_gathered

            # fill the bucket
            for i in range(num_bucket):
                if i == num_bucket - 1:
                    train_bucket[i] += (ratio >= bucket_max).sum().item()
                elif i == 0:
                    train_bucket[i] += (ratio < bucket_min).sum().item()
                train_bucket[i] += ((ratio >= bucket_min + i * bucket_size) & (ratio < bucket_min + (i + 1) * bucket_size)).sum().item()
            
            avg_loss = accelerator.gather(loss).mean()
            train_loss_meter.update(avg_loss, args.batch_size)

        if accelerator.is_main_process:
            logger.log(f"Train Loss {train_loss_meter.avg}")
            # log the bucket
            logger.log("Train Bucket:")
            for i in range(num_bucket):
                logger.log(f"Bucket range: [{bucket_min + i * bucket_size}, {bucket_min + (i + 1) * bucket_size}), Number: {train_bucket[i]}, Ratio: {train_bucket[i] / len(train_dataset)}")

        train_loss_meter.reset()

        for data in tqdm(test_dataloader):
            ply_data = data['ply'].to(device)
            ply_scale = data['ply_scale'].to(device)
            gt_extent = data['part_extent'].to(device)
            filepaths = data['path']

            estimated_extent = model(ply_data) * ply_scale
            loss = loss_fn(estimated_extent, gt_extent)

            estimated_extent_gathered = accelerator.gather(estimated_extent)
            gt_extent_gathered = accelerator.gather(gt_extent)
            filepaths_gathered = accelerator.gather(filepaths)

            # save the predicted extent
            for i in range(len(estimated_extent_gathered)):
                root = '/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset/part_scale_predicted'
                cat_root = os.path.join(root, args.part_category)
                os.makedirs(cat_root, exist_ok=True)
                with open(os.path.join(cat_root, filepaths_gathered[i].split('/')[-1].replace('.ply', '.json')), 'w') as f:
                    f.write(f'{{"scale": {estimated_extent_gathered[i].item()}}}')

            ratio = estimated_extent_gathered / gt_extent_gathered
            for i in range(num_bucket):
                if i == num_bucket - 1:
                    test_bucket[i] += (ratio >= bucket_max).sum().item()
                elif i == 0:
                    test_bucket[i] += (ratio < bucket_min).sum().item()
                test_bucket[i] += ((ratio >= bucket_min + i * bucket_size) & (ratio < bucket_min + (i + 1) * bucket_size)).sum().item()

            avg_loss = accelerator.gather(loss).mean()
            test_loss_meter.update(avg_loss, args.batch_size)

        if accelerator.is_main_process:
            logger.log(f"Test Loss {test_loss_meter.avg}")
            logger.log("Test Bucket:")
            for i in range(num_bucket):
                logger.log(f"Bucket range: [{bucket_min + i * bucket_size}, {bucket_min + (i + 1) * bucket_size}), Number: {test_bucket[i]}, Ratio: {test_bucket[i] / len(test_dataset)}")

        test_loss_meter.reset()

def save_model(model, optimizer, epoch, path):
    
    try:
        model_state_dict = model.module.state_dict()
    except:
        model_state_dict = model.state_dict()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, path)

if __name__ == "__main__":
    main()
