import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import argparse
import math
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda.amp import autocast, GradScaler

import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from model import PiLaMIM
from utils import setup_seed
import wandb 

def ddp_setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost' 
    os.environ['MASTER_PORT'] = '29871' 
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args):

    ddp_setup(rank, world_size)
    
    if rank == 0:
        wandb.init(project='pretrain', name="PiLaMIM")
        wandb.config.update(args)

    save_path = './checkpoint/' + 'pilamim' + '_' + args.model_size + '_' + args.dataset
    if rank == 0:
        os.makedirs(save_path, exist_ok=True)

    setup_seed(args.seed)

    batch_size = args.batch_size
    batch_per_device = args.per_device_batch_size
    assert batch_size % batch_per_device == 0
    steps_per_update = 1

    if args.dataset == 'imagenet1k':
        args.patch_size = 16
        args.image_size = 224
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        train_dataset = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform_train)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}.')

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_per_device, 
                                               sampler=train_sampler, 
                                               num_workers=4, 
                                               shuffle=False, 
                                               drop_last=True, 
                                               pin_memory=True)

    if args.model_size == 'base':
        model = PiLaMIM(image_size= args.image_size, patch_size= args.patch_size, emb_dim=768, decoder_emb_dim= 384, latent_decoder_emb_dim=384, encoder_layer=12, encoder_head=12, decoder_layer=6, decoder_head=12, mask_ratio=args.mask_ratio)
    
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    ema = [0.996, 1.0]
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(len(train_loader)*args.total_epoch)
                        for i in range(int(len(train_loader)*args.total_epoch)+1))

    if rank == 0:
        print("len of train dataloader: ", len(train_loader))

    scaler = GradScaler()
    save_frequency = 50
    step_count = 0

    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        train_sampler.set_epoch(e)
        losses = []
        losses_image = []
        losses_latent = []
        losses_cls = []
        for img, label in tqdm(iter(train_loader)):
            step_count += 1
            img = img.to(rank)

            optim.zero_grad()
            with autocast():
                predicted_img, mask, predict_latent, mask_latent, target_features, cls_p, cls_l, target_cls = model(img)

                loss_image = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                loss_latent = torch.mean((predict_latent - target_features) ** 2 * mask_latent) / args.mask_ratio
                loss_cls = torch.mean((target_cls - cls_l) ** 2)

                loss = loss_image + loss_latent + loss_cls

            scaler.scale(loss).backward()

            if step_count % steps_per_update == 0:
                if rank == 0:
                    print(f'In epoch {e}, step {step_count}, loss is {loss.item()}.')
                    wandb.log({"loss_iter" : loss.item(),
                            "loss_image_iter" : loss_image.item(),
                            "loss_latent_iter" : loss_latent.item(),
                            "loss_cls_iter" : loss_cls.item()}
                            , step=step_count)
                scaler.step(optim)
                scaler.update()
                
            loss = loss.detach()
            loss_image = loss_image.detach()
            loss_latent = loss_latent.detach()
            loss_cls = loss_cls.detach()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_image, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_latent, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_cls, op=dist.ReduceOp.SUM)

            loss = loss / world_size
            loss_image = loss_image / world_size
            loss_latent = loss_latent / world_size
            loss_cls = loss_cls / world_size

            losses.append(loss.item())
            losses_image.append(loss_image.item())
            losses_latent.append(loss_latent.item())
            losses_cls.append(loss_cls.item())
            
            model.module.update_target(next(momentum_scheduler)) ## target update by EMA

        lr_scheduler.step()
        if rank == 0:
            avg_loss = sum(losses) / len(losses)
            avg_loss_image = sum(losses_image) / len(losses_image)
            avg_loss_latent = sum(losses_latent) / len(losses_latent)
            avg_loss_cls = sum(losses_cls) / len(losses_cls)

            wandb.log({'learning_rate': lr_scheduler.get_last_lr()[0],
                    'loss': avg_loss,
                    'image_loss': avg_loss_image,
                    'latent_loss': avg_loss_latent,
                    'cls_loss': avg_loss_cls,
                    }, step=step_count)
            
            print(f'In epoch {e}, average traning loss is {avg_loss}.')

            ''' save model '''
            if ((e+1) % save_frequency == 0 or (e+1) == args.total_epoch) and rank == 0:
                torch.save(model.module.state_dict(), save_path + "-"+ str(e) +".pt")
    
    if rank == 0:
        wandb.finish()
    destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--per_device_batch_size', type=int, default=512)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=800)
    parser.add_argument('--warmup_epoch', type=int, default=40)
    parser.add_argument('--dataset', type=str, default='imagenet1k')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--patch_size', type=int, default = 16)
    parser.add_argument('--image_size', type=int, default = 224)

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(f'world_size: {world_size}')

    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
