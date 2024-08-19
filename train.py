import argparse
from model.mdsam import MDSAM
from dataset.sod_dataset import getSODDataloader
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
from collections import OrderedDict
import torch.distributed as dist
import time
from utils.loss import LossFunc
from utils.AvgMeter import AvgMeter

dist.init_process_group(backend="nccl")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--seed", type=int, default = 42
    )
    parser.add_argument(
        "--warmup_period", type = int, default = 5,    
    )
    parser.add_argument(
        "--batch_size", type = int, default = 1, required = True    
    )
    parser.add_argument(
        "--num_workers", type = int, default = 0
    )
    parser.add_argument(
        "--epochs", type = int, default=80
    )
    parser.add_argument(
        "--lr_rate", type = float, default = 0.0005,
    )
    parser.add_argument(
        "--img_size", type = int, default = 512
    )
    parser.add_argument(
        "--data_path", type = str, required = True, help="the postfix must to be DUTS"
    )
    parser.add_argument(
        "--sam_ckpt", type = str
    )
    parser.add_argument(
        "--save_dir", type = str, default = "output/"
    )
    parser.add_argument(
        "--resume", type = str, default = "", help="If you need to train from begining, make sure 'resume' is empty str. If you want to continue training, set it to the previous checkpoint."
    )
    parser.add_argument(
        "--local-rank", type=int, default=-1, help="For distributed training: local_rank"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def trainer(net, dataloader, loss_func, optimizer, local_rank):
    net.train()
    loss_avg = AvgMeter()
    mae_avg = AvgMeter()
    if local_rank == 0:
        print("start trainning")
    
    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    if local_rank == 0:
        data_generator = tqdm(dataloader)
    else:
        data_generator = dataloader
    for data in data_generator:
        img = data["img"].to(device).to(torch.float32)
        label = data["mask"].to(device).unsqueeze(1)
        
        optimizer.zero_grad()

        out, coarse_out = net(img)

        out = sigmoid(out)
        coarse_out = sigmoid(coarse_out)

        loss_out = loss_func(out, label)
        loss_coarse = loss_func(coarse_out, label)

        loss = loss_out + loss_coarse
        
        loss_avg.update(loss.item(), img.shape[0])

        img_mae = torch.mean(torch.abs(out - label))

        mae_avg.update(img_mae.item(),n=img.shape[0])

        loss.backward()
        optimizer.step()
    
    temp_cost=time.time()-start
    print("local_rank:{}, loss:{}, mae:{}, cost_time:{:.0f}m:{:.0f}s".format(local_rank, loss_avg.avg, mae_avg.avg, temp_cost//60, temp_cost%60))
 
def valer(net, dataloader, local_rank):
    net.eval()
    if local_rank == 0:
        print("start valling")

    start = time.time()

    sigmoid = torch.nn.Sigmoid()
    mae_avg = AvgMeter()
    with torch.no_grad():
        if local_rank == 0:
            data_generator = tqdm(dataloader)
        else:
            data_generator = dataloader
        for data in data_generator:
                
            img = data["img"].to(device).to(torch.float32)
            ori_label = data['ori_mask'].to(device)
            
            out, coarse_out= net(img)
            out = sigmoid(out)
            out = torch.nn.functional.interpolate(out, [ori_label.shape[1],ori_label.shape[2]], mode = 'bilinear', align_corners = False)

            #Since the float values are converted to int when saving the mask, 
            #multiple decimal will be lost, which may result in minor deviations from the evaluation code.
            img_mae=torch.mean(torch.abs(out - ori_label))

            mae_avg.update(img_mae.item(),n=1)
    
    temp_cost=time.time() - start

    print("local_rank:{}, val_mae:{}, cost_time:{:.0f}m:{:.0f}s".format(local_rank, mae_avg.avg, temp_cost//60, temp_cost%60))

    return mae_avg.avg



def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in  k or '8' in k or '11' in k):
        return rel_pos_params
    
    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]

def load(net,ckpt, img_size):
    ckpt=torch.load(ckpt,map_location='cpu')
    from collections import OrderedDict
    dict=OrderedDict()
    for k,v in ckpt.items():
        #把pe_layer改名
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k :
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            if "neck" in k:
                #Add the original final neck layer to 3, 6, and 9, initialization is the same.
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict[new_key] = v
            else:
                dict[k]=v
        if "mask_decoder.transformer" in k:
            dict[k] = v
        if "mask_decoder.iou_token" in k:
            dict[k] = v
        if "mask_decoder.output_upscaling" in k:
            dict[k] = v
    state = net.load_state_dict(dict, strict=False)
    return state

if __name__ == "__main__":

    args = parse_args()
    if args.local_rank == 0:
        print("start training, batch_size: {}, lr_rate: {}, warmup_period: {}, save to {}".format(args.batch_size, args.lr_rate, args.warmup_period, args.save_dir))
    torch.manual_seed(args.seed)
    device = torch.device(f"cuda:{args.local_rank}")

    #Model definition and loading SAM pre-trained weights
    net = MDSAM(args.img_size).to(device)
    if args.resume == "":
        state = load(net, args.sam_ckpt, args.img_size)
        if args.local_rank == 0:
            print(state)

    trainLoader = getSODDataloader(args.data_path, args.batch_size, args.num_workers, 'train', img_size= args.img_size)
    valLoader = getSODDataloader(args.data_path, 1, args.num_workers, 'test', args.local_rank, img_size= args.img_size, max_rank = dist.get_world_size())

    loss_func = LossFunc

    #freeze layers and define different lr_rate for layers
    hungry_param = []
    full_param = []
    for k,v in net.named_parameters():
        if "image_encoder" in k:
            if "adapter" in k:
                hungry_param.append(v)
            elif "neck" in k:
                full_param.append(v)
            else:
                v.requires_grad = False
        else:
            if "transformer" in k:
                full_param.append(v)
            elif "iou" in k:
                full_param.append(v)
            elif "mask_tokens" in k:
                hungry_param.append(v)
            elif "pe_layer" in k:
                full_param.append(v)
            elif "output_upscaling" in k:
                full_param.append(v)
            else:
                hungry_param.append(v)      
    
    optimizer = torch.optim.AdamW([{"params": hungry_param, "lr": args.lr_rate}, {"params" : full_param, "lr" : args.lr_rate * 0.1}], weight_decay=1e-5)
    
    best_mae = 1
    best_epoch = 0

    start_epoch = 1

    #resume from checkpoint
    if args.resume != "":
        start_epoch = int(args.resume.split("/")[-1].split(".")[0][11:]) + 1
        resume_dict = torch.load(args.resume, map_location= "cpu")
        optimizer.load_state_dict(resume_dict["optimizer"])
        net_dict = OrderedDict()

        for k,v in resume_dict['model'].items():
            if "module." in k:
                net_dict[k[7:]] = v
            else:
                net_dict[k] = v
        state = net.load_state_dict(net_dict)
        if args.local_rank == 0:
            print(state)

    net=torch.nn.parallel.DistributedDataParallel(net.to(device),device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)

    for i in range(start_epoch, args.epochs + 1):

        #lr_rate setting
        if i <= args.warmup_period:
            _lr = args.lr_rate * i / args.warmup_period
        else:
            _lr = args.lr_rate * (0.98 ** (i - args.warmup_period))

        t = 0
        for param_group in optimizer.param_groups:
            if t == 0:
                param_group['lr'] = _lr
            else:
                param_group['lr'] = _lr * 0.1
            t += 1
        
        if args.local_rank == 0:
            print("epochs {} start".format(i))

        trainer(net, trainLoader, loss_func, optimizer, local_rank=args.local_rank)

        local_mae = valer(net, valLoader, local_rank = args.local_rank)

        #average the results from multi-GPU inference
        sum_result = torch.tensor(local_mae).to(device)
        dist.reduce(sum_result, dst = 0, op = dist.ReduceOp.SUM)

        if args.local_rank == 0:
            mae = sum_result.item() / dist.get_world_size()
            print("current mae:{}".format(mae))
            #save the best result
            if(mae < best_mae):
                best_mae = mae
                best_epoch = i
                print("save epoch {} in {}".format(i, "{}/model_epoch{}.pth".format(args.save_dir,i)))
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save({"model": net.state_dict(),"optimizer":optimizer.state_dict()}, "{}/model_epoch{}.pth".format(args.save_dir,i))
            print("best epoch:{}, mae:{}".format(best_epoch,best_mae))
        






