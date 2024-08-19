import argparse
from dataset.sod_dataset import getSODDataloader
from model.mdsam import MDSAM
import torch
from tqdm import tqdm
import os
import shutil
from collections import OrderedDict
import numpy as np
import cv2
import os
import py_sod_metrics

datasets = ["DUTS", "PASCAL-S", "DUT-OMRON", "ECSSD", "HKU-IS", ]

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a evaluation script.")
    parser.add_argument(
        "--checkpoint", type = str, required = True,    
    )
    parser.add_argument(
        "--data_path", type = str, required = True    
    )
    parser.add_argument(
        "--result_path", type = str, default = "./output"
    )
    parser.add_argument(
        "--num_workers", type = int, default = 0
    )
    parser.add_argument(
        "--img_size", type = int, default = 512,
    )
    parser.add_argument(
        "--gpu_id", type = int, default = 0
    )

    args = parser.parse_args()

    return args

def eval(net, dataloader, output_path, dataset):
    net.eval()
    print("start eval dataset:{}".format(dataset))

    sigmoid = torch.nn.Sigmoid()

    MAE = py_sod_metrics.MAE()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    FM = py_sod_metrics.Fmeasure()
    
    with torch.no_grad():
        for data in tqdm(dataloader, ncols= 100):
            
            img = data["img"].to(device).to(torch.float32)
            ori_label = data['ori_mask']
            name = data['mask_name']
            
            out, corase_out = net(img)
            out = sigmoid(out)
            out = torch.nn.functional.interpolate(out, [ori_label.shape[1],ori_label.shape[2]], mode = 'bilinear', align_corners = False)

            pred = (out * 255).squeeze().cpu().data.numpy().astype(np.uint8)
            ori_label = (ori_label * 255).squeeze(0).data.numpy().astype(np.uint8)

            FM.step(pred=pred, gt=ori_label)
            WFM.step(pred=pred, gt=ori_label)
            SM.step(pred=pred, gt=ori_label)
            EM.step(pred=pred, gt=ori_label)
            MAE.step(pred=pred, gt=ori_label)

            pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(output_path + "/" + name[0], pred) 
    
    fm = FM.get_results()["fm"]
    pr = FM.get_results()["pr"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]



    maxFm = FM.get_results()['mf']
    meanFm = fm['curve'].mean()
    fm = fm['adp']
    em = em['curve'].mean()

    print("{} results:".format(dataset))
    print("mae:{:.3f}, maxFm:{:.3f},  sm:{:.3f}, em:{:.3f}".format(mae, maxFm,sm, em))
    

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}")
    net = MDSAM(args.img_size).to(device)
    #load pretrained weights
    ckpt_dic = torch.load(args.checkpoint)
    if 'model' in ckpt_dic.keys():
        ckpt_dic = ckpt_dic['model']
    dic = OrderedDict()
    for k,v in ckpt_dic.items():
        if 'module.' in k:
            dic[k[7:]] = v
        else:
            dic[k] = v
    msg = net.load_state_dict(dic, strict = False)
    print(msg)

    datasets = ["DUTS", "PASCAL-S", "DUT-OMRON", "ECSSD", "HKU-IS"] #["SOC","SOC-AC","SOC-BO","SOC-CL","SOC-HO","SOC-MB","SOC-OC","SOC-OV","SOC-SC","SOC-SO"]
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    for dataset in datasets:
        testLoader = getSODDataloader(os.path.join(args.data_path, dataset), 1, args.num_workers, 'test', 0, max_rank= 1, img_size= args.img_size)

        dataset_result_path = os.path.join(args.result_path, dataset)

        if os.path.exists(dataset_result_path):
            shutil.rmtree(dataset_result_path)
        os.makedirs(dataset_result_path)

        eval(net, testLoader, dataset_result_path, dataset)


    