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
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--gpu_ids", type=str, default="0")
parser.add_argument("--experiment_name", type=str, required=True)
parser.add_argument("--part_category", type=str, default="slider_drawer")
parser.add_argument(
    "--dataroot",
    type=str,
    default="/raid/haoran/Project/PartDiffusion/PartDiffusion/dataset",
)
parser.add_argument("--ply_rotate", action="store_true", help="rotate the ply files")
parser.add_argument("--extend_size_train", type=int, default=None, help="extend the dataset size")
parser.add_argument("--extend_size_test", type=int, default=None, help="extend the dataset size")

parser.add_argument(
    "--hidden_dim", type=int, default=256, help="hidden dim of PointNet++"
)
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

parser.add_argument(
    "--load_pretrain", action='store_true', help="load pretrain model"
)
parser.add_argument(
    "--cond_ckpt",
    type=str,
    default="/raid/haoran/Project/PartDiffusion/PartDiffusion/pretrained_checkpoint/pointnet2.pth",
    help="condition model ckpt to load.",
)
parser.add_argument('--loss_fn', type=str, default='l2', help='loss function', choices=['l1', 'l2'])

parser.add_argument("--continue_train", action="store_true", help="continue training")

parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
parser.add_argument("--num_epochs", type=int, default=40)

args = parser.parse_args()

def main():

    experiment_path = os.path.join('./logs_scale_pred', args.experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    logger = Logger(os.path.join(experiment_path, 'log.txt'))

    model = PointNet4ScalePrediction()
    if args.load_pretrain and not args.continue_train:
        load_result = model.load_state_dict(
            torch.load(args.cond_ckpt)["model_state_dict"], strict=False
        )
        logger.log(str(load_result))
        logger.log('conditional model successfully loaded')

    if args.continue_train:
        pretrained_weight = torch.load(os.path.join(experiment_path, 'checkpoint/model_latest.pth'))
        model.load_state_dict(pretrained_weight['model_state_dict'])
        load_epoch = pretrained_weight['epoch']
        logger.log(f'Continue training from epoch {load_epoch}')

    train_dataset, test_dataset = GAPartNetDataset4ScalePrediction(
        args, phase="train", cat=args.part_category, extend_size=args.extend_size_train
    ), GAPartNetDataset4ScalePrediction(args, phase="test", cat=args.part_category, extend_size=args.extend_size_test)

    train_dataloader, test_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    ), DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10*len(train_dataloader), gamma=0.1)

    accelerator = Accelerator()
    device = accelerator.device

    model, optimizer, scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    loss_fn = nn.functional.l1_loss if args.loss_fn == 'l1' else nn.functional.mse_loss

    train_loss_meter, test_loss_meter = AverageMeter(), AverageMeter()
    train_loss_meter.reset()
    test_loss_meter.reset()

    # Main train loop
    for epoch in range(args.num_epochs):

        model.train()
        for data in tqdm(train_dataloader):
            
            ply_data = data['ply'].to(device)
            ply_scale = data['ply_scale'].to(device)
            gt_extent = data['part_extent'].to(device)

            estimated_extent = model(ply_data) * ply_scale
            # print(scale.shape, estimated_scale.shape)
            loss = loss_fn(estimated_extent, gt_extent)

            avg_loss = accelerator.gather(loss).mean()
            train_loss_meter.update(avg_loss, args.batch_size)

            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        if accelerator.is_main_process:
            logger.log(f"Epoch {epoch}, Train Loss {train_loss_meter.avg}")
        train_loss_meter.reset()

        # test
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                ply_data = data['ply'].to(device)
                ply_scale = data['ply_scale'].to(device)
                gt_extent = data['part_extent'].to(device)

                estimated_extent = model(ply_data) * ply_scale
                loss = loss_fn(estimated_extent, gt_extent)

                avg_loss = accelerator.gather(loss).mean()
                test_loss_meter.update(avg_loss, args.batch_size)

        if accelerator.is_main_process:
            logger.log(f"Epoch {epoch}, Test Loss {test_loss_meter.avg}")
        test_loss_meter.reset()

        # save model checkpoint
        save_model(model, optimizer, epoch, os.path.join(experiment_path, 'checkpoint/model_latest.pth'))
        if epoch % args.save_freq == 0:
            save_model(model, optimizer, epoch, os.path.join(experiment_path, f'checkpoint/model_{epoch}.pth'))

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
