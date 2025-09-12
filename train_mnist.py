import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
import matplotlib.pyplot as plt
from utils import ExponentialMovingAverage
import pdb
import os
import math
import argparse

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--predict_idx', type=int, help = 'load model and predict the the probability')
    args = parser.parse_args()

    return args

def evaluate(model, test_dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for image, target in test_dataloader:
            image = image.to(device)
            binary_target = (target == 0).float().to(device)  # 1 if digit=0 else 0

            # forward pass
            pred = model(image, torch.randn_like(image).to(device))
            logits = pred[:, 0, 0, 0]  # using [0,0] pixel as logit
            probs = torch.sigmoid(logits)
            pred_cls = (probs > 0.5).float()

            # accumulate
            total_correct += (pred_cls == binary_target).sum().item()
            total_samples += binary_target.size(0)

    return total_correct / total_samples

def predict_is_zero(model, device, idx=0, noise_std=0.2):
    """
    Take one MNIST test sample, add noise, and predict whether it's '0' or not.
    """
    # load MNIST test set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    test_dataset = MNIST(root="./mnist_data", train=False, download=True, transform=transform)

    # get one image + label
    image, target = test_dataset[idx]
    image = image.unsqueeze(0).to(device)  # [1,1,28,28]
    label = int(target)

    # add Gaussian noise to the image
    noisy_image = image + noise_std * torch.randn_like(image)
    noisy_image = torch.clamp(noisy_image, -1, 1)  # keep in [-1,1]

    # forward pass
    model.eval()
    with torch.no_grad():
        pred = model(noisy_image, torch.randn_like(noisy_image).to(device))
        logit = pred[:, 0, 0, 0]  # [0,0] pixel as logit
        prob = torch.sigmoid(logit).item()
        is_zero = prob > 0.5

    # show
    plt.imshow(noisy_image.squeeze().cpu(), cmap="gray")
    plt.title(f"True Label: {label}, Predicted: {'0' if is_zero else 'not 0'} (p={prob:.3f})")
    plt.axis("off")
    save_path = f"./results/preds/{args.predict_idx}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved prediction image to {save_path}")

    return is_zero, prob, label

def main(args):
    device="cpu" if args.cpu else "cuda"
    print(device)
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_mse_fn = nn.MSELoss(reduction='mean')
    loss_bce_fn = nn.BCEWithLogitsLoss()

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    if args.predict_idx:
        predict_is_zero(model, device, idx=args.predict_idx, noise_std=0.3)
        return

    global_steps=0
    for i in range(args.epochs):
        model.train()
        for j,(image,target) in enumerate(train_dataloader):
            noise=torch.randn_like(image).to(device)
            mask = torch.ones_like(image).to(device)
            image=image.to(device)
            pred=model(image,noise)
            # [0,0] pixel of image is used as logit prediction.
            logits = pred[:, 0, 0, 0]
            binary_target = (target == 0).float().to(device)  # shape [B]
            # mask for noise prediction to ignore [0,0] pixel of image, which is used for prediction.
            mask[:,0,0,0] = 0
            loss_bce = loss_bce_fn(logits, binary_target)
            # using both mse and cl loss.
            loss_mse = loss_mse_fn(pred*(mask),noise*(mask))
            loss = loss_bce + loss_mse
            # # with already trained model, and just finetuning for classification.
            # with torch.no_grad():
            #     loss_mse = loss_mse_fn(pred*(mask),noise*(mask))
            # loss = loss_bce
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss_bce:{:.5f},loss_mse:{:.5f},loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss_bce.detach().cpu().item(),loss_mse.detach().cpu().item(),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
                test_acc = evaluate(model, test_dataloader, device)
                print(f"Test Accuracy: {test_acc:.4f}")
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}

        os.makedirs("results",exist_ok=True)
        torch.save(ckpt,"results/steps_{:0>8}.pt".format(global_steps))
        model_ema.eval()
        samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        save_image(samples,"results/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(args.n_samples)))


if __name__=="__main__":
    args=parse_args()
    main(args)