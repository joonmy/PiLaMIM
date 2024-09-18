import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import math
import torch

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.cuda.amp import autocast, GradScaler

from lars import LARS
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from model import *
from utils import setup_seed

import wandb 

def ddp_setup(rank, world_size):

    os.environ['MASTER_ADDR'] = 'localhost' # single machine setup
    os.environ['MASTER_PORT'] = '19487' 
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

def main(rank, world_size, args):
    ddp_setup(rank, world_size)

    if rank == 0:
        wandb.init(project='linear_probing', name=args.dataset)
        wandb.config.update(args)

    save_path = "./linear_probing_checkpoint/" 

    if rank == 0:
        os.makedirs(save_path, exist_ok=True)

    setup_seed(args.seed)

    batch_size = args.batch_size
    batch_per_device = args.per_device_batch_size
    assert batch_size % batch_per_device == 0
    steps_per_update = 1

    args.patch_size = 16
    args.image_size = 224
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        num_classes = 10

    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.CIFAR100('./data', train=True, download=True, transform=transform)
        val_dataset = torchvision.datasets.CIFAR100('./data', train=False, download=True, transform=transform)
        num_classes = 100

    elif args.dataset == 'imagenet1k':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.ImageNet(root='./data', split='train', transform=transform_train)
        val_dataset = torchvision.datasets.ImageNet(root='./data', split='val', transform=transform_val)
        num_classes = 1000

    elif args.dataset == 'Places365':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            # transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.Places365(root='./data', split='train-standard', small = True, download = True, transform=transform_train)
        val_dataset = torchvision.datasets.Places365(root='./data', split='val', small = True, download = True, transform=transform_val)
        num_classes = 365

    elif args.dataset == 'INat2021':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.INaturalist(root='./data', version='2021_train_mini', download=True, transform=transform_train)
        val_dataset = torchvision.datasets.INaturalist(root='./data', version='2021_valid', download=True, transform=transform_val)
        num_classes = 10000

    elif args.dataset == 'CLEVR':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.CLEVRClassification(root='./data', split='train', download=True,  transform=transform_train, target_transform=None)
        val_dataset = torchvision.datasets.CLEVRClassification(root='./data', split='val', download=True,  transform=transform_val, target_transform=None)
        num_classes = 8 # label : min 3 max 10,  so, label = label - 3  (0 ~ 7)

    elif args.dataset == 'CLEVR_Dist':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_val = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        train_dataset = torchvision.datasets.CLEVRClassification(root='./data', split='train', download=True, dist = True,  transform=transform_train, target_transform=None)
        val_dataset = torchvision.datasets.CLEVRClassification(root='./data', split='val', download=True,  dist = True, transform=transform_val, target_transform=None)
        num_classes = 6 

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
    
    valid_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_per_device,
                                            sampler=valid_sampler,
                                            num_workers=4,
                                            shuffle=False,
                                            drop_last=False,
                                            pin_memory=True)

    model_name = args.pretrained_model_path.split('/')[-1]
    
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
    
    if args.pretrained_model_path is not None:
        model = PiLaMIM(image_size= args.image_size, patch_size= args.patch_size, emb_dim=768, decoder_emb_dim= 384, latent_decoder_emb_dim=384, encoder_layer=12, encoder_head=12, decoder_layer=6, decoder_head=12)
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location='cpu'))
        
        for param in model.encoder.parameters():
            param.requires_grad = False

        model = ViT_Classifier(model.encoder, num_classes=num_classes)

        args.output_model_path = (args.dataset + model_name)
    
    if rank == 0:
        for param, name in zip(model.parameters(), model.state_dict().keys()):
            print(name, param.requires_grad)
        
        print(args.pretrained_model_path)

    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = LARS(model.parameters(), lr=3, momentum=0.9)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    if rank == 0:
        print("len of train dataloader: ", len(train_loader))
        print("len of valid dataloader: ", len(valid_loader))
    
    scaler = GradScaler()
    
    best_val_acc = 0
    step_count = 0

    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        train_sampler.set_epoch(e)
        losses = []
        acces = []
        for img, label in tqdm(iter(train_loader)):
            step_count += 1
            img = img.to(rank)
            label = label.to(rank)

            if args.dataset == 'CLEVR':
                label = label - 3

            optim.zero_grad()
            with autocast():
                logits = model(img)
                loss = loss_fn(logits, label)

            scaler.scale(loss).backward()

            if step_count % steps_per_update == 0:
                if rank == 0:
                    print(f'In epoch {e}, step {step_count}, loss is {loss.item()}')
                    wandb.log({"loss_iter": loss.item()}, step=step_count)
                scaler.step(optim)
                scaler.update()

            acc = acc_fn(logits, label)

            loss = loss.detach()
            acc = acc.detach()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc, op=dist.ReduceOp.SUM) 
            loss = loss / world_size
            acc = acc / world_size
    
            losses.append(loss.item())
            acces.append(acc.item())

        lr_scheduler.step()
        if rank == 0:
            avg_train_loss = sum(losses) / len(losses)
            avg_train_acc = sum(acces) / len(acces)
            
            wandb.log({"train_loss": avg_train_loss, "train_acc": avg_train_acc, "learning_rate": lr_scheduler.get_last_lr()[0]}, step=step_count)
            print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(valid_loader)):
                img = img.to(rank)
                label = label.to(rank)

                if args.dataset == 'CLEVR':
                    label = label - 3

                with autocast():
                    logits = model(img)
                    loss = loss_fn(logits, label)

                acc = acc_fn(logits, label)

                loss = loss.detach()
                acc = acc.detach()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(acc, op=dist.ReduceOp.SUM)
                loss = loss / world_size
                acc = acc / world_size

                losses.append(loss.item())
                acces.append(acc.item())

            if rank == 0:
                avg_val_loss = sum(losses) / len(losses)
                avg_val_acc = sum(acces) / len(acces)
                wandb.log({"val_loss": avg_val_loss, "val_acc": avg_val_acc}, step=step_count)
                print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    print(f'saving best model with acc {best_val_acc} at {e} epoch!')       
                    torch.save(model, save_path + args.output_model_path)
    if rank == 0:
        wandb.finish()
    destroy_process_group()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--per_device_batch_size', type=int, default= 1024)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=10) 
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='')
    parser.add_argument('--dataset', type=str, default='imagenet1k')

    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    print(f'world_size: {world_size}')

    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)

   