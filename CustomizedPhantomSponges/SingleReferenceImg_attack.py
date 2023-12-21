import json
import pickle
import random
import time
from pathlib import Path
import torch
import numpy as np
import torchvision
from torchvision import transforms  # 为了保存 patch
import torch.nn as nn
import shutil
import cv2

# 计算 benign_detection_loss 的时候，使用的 nms 算法; 使用phantom提供的 inplace-free 版本的nms
from CustomizedPhantomSponges.utils import *

workdir = Path(r'G:\NMSProject\CustomizedPhantomSponges\workdir')
pretrainedCheckpointDir = Path(r'G:\NMSProject\CustomizedPhantomSponges\pretrainedCheckpoints')

trailDir = workdir / 'tmp_trail_XXX'
trailDir.mkdir(exist_ok=True, parents=True)


def get_model(name, ckpt):
    # 控制使用 本地 的 model结构 以及 model的ckpt
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('[Info] device = {}'.format(device))
    
    if name == 'yolov5':
        from PhantomSponges.local_yolos.yolov5.models.experimental import attempt_load
        model = attempt_load(ckpt, device).eval()
        
    # elif name == 'yolov4':
    #     # taken from https://github.com/WongKinYiu/PyTorch_YOLOv4
    #     from local_yolos.yolov4.models.models import Darknet, load_darknet_weights
    #     model = Darknet('local_yolos/yolov4/cfg/yolov4.cfg', img_size=640).to(device).eval()
    #     load_darknet_weights(model, 'local_yolos/yolov4/weights/yolov4.weights')
    # elif name == 'yolov3':
    #     # taken from https://github.com/ultralytics/yolov3
    #     from local_yolos.yolov3 import hubconf
    #     model = hubconf.yolov3(pretrained=True, autoshape=False, device=device)
    return model


def compute_max_objects_loss(output_patch, target_class=0, conf_thres=0.25):
    # strategy 1
    # ToDo target class 2-> car; 0-> person
    # see Daedalus_attack_master\data\coco_classes.txt
    
    x2 = output_patch[:, :, 5:] * output_patch[:, :, 4:5]
    
    conf, j = x2.max(2, keepdim=False)
    all_target_conf = x2[:, :, target_class]
    under_thr_target_conf = all_target_conf[conf < conf_thres]
    
    conf_avg = len(conf.view(-1)[conf.view(-1) > conf_thres]) / len(output_patch)
    # print(f"pass to NMS: {conf_avg}")
    
    zeros = torch.zeros(under_thr_target_conf.size()).to(output_patch.device)
    zeros.requires_grad = True
    x3 = torch.maximum(-under_thr_target_conf + conf_thres, zeros)
    mean_conf = torch.sum(x3, dim=0) / (output_patch.size()[0] * output_patch.size()[1])
    
    return mean_conf, conf_avg


def compute_bboxes_area_loss(output_patch, patch_size, conf_thres=0.25):
    # strategy 2.a
    
    t_loss = 0.0
    preds_num = 0
    
    xc_patch = output_patch[..., 4] > conf_thres
    not_nan_count = 0
    
    # [Note] zyx: this loss func does not use output_clean
    # For each img in the batch
    # for (xi, x), (li, l) in (zip(enumerate(output_patch), enumerate(output_clean))):  # image index, image inference
    for (xi, x) in enumerate(output_patch):
        x1 = x[xc_patch[xi]]  # .clone()
        x2 = x1[:, 5:] * x1[:, 4:5]  # x1[:, 5:] *= x1[:, 4:5]
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box_x1 = xywh2xyxy(x1[:, :4])
        
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        agnostic = True
        
        conf_x1, j_x1 = x2.max(1, keepdim=True)
        x1_full = torch.cat((box_x1, conf_x1, j_x1.float()), 1)[conf_x1.view(-1) > conf_thres]
        c_x1 = x1_full[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes_x1, scores_x1 = x1_full[:, :4] + c_x1, x1_full[:, 4]  # boxes (offset by class), scores
        final_preds_num = len(torchvision.ops.nms(boxes_x1, scores_x1, conf_thres))
        preds_num += final_preds_num
        
        # calculate bboxes' area avg
        bboxes_x1_wh = xyxy2xywh(boxes_x1)[:, 2:]
        bboxes_x1_area = bboxes_x1_wh[:, 0] * bboxes_x1_wh[:, 1]
        img_loss = bboxes_x1_area.mean() / (patch_size[0] * patch_size[1])
        if not torch.isnan(img_loss):
            t_loss += img_loss
            not_nan_count += 1
    
    if not_nan_count == 0:
        t_loss_f = torch.tensor(torch.nan)
    else:
        t_loss_f = t_loss / not_nan_count
    
    return t_loss_f


def compute_benign_detection_loss(output_clean, output_patch, img_size, device, conf_threshold, iou_threshold,
                                  patch_conf_threshold):
    batch_loss = []
    
    gn = torch.tensor(img_size)[[1, 0, 1, 0]]
    gn = gn.to(device)
    # 原始yolo里面的nms实现，是有 inplace 操作的，因此要改成inplace_free 版本
    # pred_clean_bboxes = non_max_suppression(output_clean, conf_threshold, iou_threshold, classes=None, max_det=1000)
    pred_clean_bboxes = non_max_suppression_inplace_free(output_clean, conf_threshold, iou_threshold, classes=None,
                                                         max_det=1000)
    # pred_patch_bboxes = non_max_suppression(output_patch, patch_conf_threshold, iou_threshold, classes=None, max_det=30000)
    pred_patch_bboxes = non_max_suppression_inplace_free(output_patch, patch_conf_threshold, iou_threshold,
                                                         classes=None, max_det=30000)
    
    # print final amount of predictions
    final_preds_batch = 0
    # for img_preds in non_max_suppression(output_patch, conf_threshold, iou_threshold, classes=None, max_det=30000):
    for img_preds in non_max_suppression_inplace_free(output_patch, conf_threshold, iou_threshold, classes=None,
                                                      max_det=30000):
        final_preds_batch += len(img_preds)
    
    for (img_clean_preds, img_patch_preds) in zip(pred_clean_bboxes, pred_patch_bboxes):  # per image
        
        for clean_det in img_clean_preds:
            
            clean_clss = clean_det[5]
            
            clean_xyxy = torch.stack([clean_det])  # .clone()
            clean_xyxy_out = (clean_xyxy[..., :4] / gn).to(device)
            
            img_patch_preds_out = img_patch_preds[img_patch_preds[:, 5].view(-1) == clean_clss]
            
            patch_xyxy_out = (img_patch_preds_out[..., :4] / gn).to(device)
            
            if len(clean_xyxy_out) != 0:
                target = get_iou(patch_xyxy_out, clean_xyxy_out)
                if len(target) != 0:
                    target_m, _ = target.max(dim=0)
                else:
                    target_m = torch.zeros(1).to(device)
                
                batch_loss.append(target_m)
    
    one = torch.tensor(1.0).to(device)
    if len(batch_loss) == 0:
        return one
    
    return (one - torch.stack(batch_loss).mean())


# 基础版本，model只有一个；对一帧输入进行phantom生成
class PhantomGenerator:
    def __init__(self, model_name, model_ckpt_path, max_iter, use_cuda, conf_threshold, iou_threshold,
                 patch_conf_threshold):
        """
        model_name: yolov5 / ..
        model_ckpt_path: path to the checkpoint file
        victim_img_path: path to the benign image
        max_iter: max iteration for performing PGD ~ 10k scale
        """
        
        # 定义一些参数
        self.max_iter = max_iter
        # self.use_cuda = use_cuda
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold  # 进行 nms 的参数
        self.iou_threshold = iou_threshold  # 进行 nms 的参数
        self.patch_conf_threshold = patch_conf_threshold  # 生成 perturbation时，使用的conf (先留下更多的box)
        
        # 定义一些存储loss 记录的数组和变量
        self.max_objects_loss_list = []
        self.bboxes_area_loss_list = []
        self.benign_detection_loss_list = []
        self.train_loss_list = []
        self.pass_to_NMS_list = []
        self.time = []
        
        # 加载 model，以及权重文件
        self.model = get_model(model_name, model_ckpt_path)
        
        # # 加载victim图片
        # # ref 参考了yolov5 的img处理 \utils\dataloaders.py
        # im = cv2.imread(victim_img_path)  # HWC(BGR)
        # im = im.astype(np.float32) / 255
        # im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # self.victim_img = np.ascontiguousarray(im)  # contiguous in memory (解释: https://zhuanlan.zhihu.com/p/59767914)
        # self.patch_size_H = self.victim_img.shape[1]
        # self.patch_size_W = self.victim_img.shape[2]
        
        self.victim_img = np.zeros((3, 640, 640), dtype=np.float32)  # ToDo: single reference img
        self.patch_size_H = 640
        self.patch_size_W = 640
    
    # 某个loss func
    def loss_func(self, loss_hypers, target_class, patch_size, device, conf_threshold, iou_threshold,
                  patch_conf_threshold, output_clean, output_patch):
        # loss_hypers:
        #   lambda1 -> # obj (0.15 level)
        #   lambda2 -> IoU between boxes
        #       lambda2a -> box area  (0.03 level)
        #       lambda2b ->
        #   lambda3 -> retain original detection
        
        # output_clean & output_patch : (batch=1, 25200, 85)
        
        max_objects_loss, pass_to_NMS = compute_max_objects_loss(output_patch, target_class)
        bboxes_area_loss = compute_bboxes_area_loss(output_patch, patch_size)
        # benign_detection_loss = compute_benign_detection_loss(output_clean, output_patch, patch_size, device, conf_threshold, iou_threshold, patch_conf_threshold)
        
        loss = max_objects_loss * loss_hypers['lambda_1']
        self.max_objects_loss_list.append(loss_hypers['lambda_1'] * max_objects_loss.item())
        self.pass_to_NMS_list.append(pass_to_NMS)
        
        print('[Log Loss] max_obj={:.4f}, pass_to_NMS={:.1f}'.format(
            self.max_objects_loss_list[-1], self.pass_to_NMS_list[-1]), end=', '
        )
        # bbox_area={:.4f}, train_loss={:.4f}
        # self.bboxes_area_loss_list[-1],
        #             self.train_loss_list[-1]
        
        # For the very first few iters, bboxes_are_loss may be nan
        if not torch.isnan(bboxes_area_loss):
            loss += (bboxes_area_loss * loss_hypers['lambda_2a'])
            self.bboxes_area_loss_list.append(bboxes_area_loss.item() * loss_hypers['lambda_2a'])
            print('bbox_area={:.4f}'.format(self.bboxes_area_loss_list[-1]), end=', ')
        else:
            self.bboxes_area_loss_list.append(np.nan)
            print('bbox_area={:.4f}'.format(self.bboxes_area_loss_list[-1]), end=', ')
        
        # [Note] we do not care this loss
        # if not torch.isnan(benign_detection_loss):
        #     loss += (benign_detection_loss * loss_hypers['lambda_3'])
        #     self.benign_detection_loss_list.append(loss_hypers['lambda_3'] * benign_detection_loss.item())
        
        self.train_loss_list.append(loss.item())
        print('train_loss={:.4f}'.format(self.train_loss_list[-1]))
        
        return loss
    
    # 攻击过程
    def generate_phantom(self, loss_hypers, target_class, pgd_epsilon, fgsm_epsilon):
        start = time.time()
        
        # init perturbation to 0 (i.e., adv_patch)
        patch = torch.zeros([3, self.patch_size_H, self.patch_size_W], device=self.device)  # img in torch: C(RGB), H, W
        patch.requires_grad = True
        
        adv_patch = patch
        
        victim_img = torch.tensor(self.victim_img, device=self.device)
        # expand victim 成 N=1, C, H, W ; 因为 torch model 输入要求
        victim_img = torch.unsqueeze(victim_img, 0)
        
        torch.autograd.set_detect_anomaly(True)
        
        for curr_iter in range(1, self.max_iter):
            self.time.append(time.time() - start)
            
            applied_patch = torch.clamp(victim_img[:] + adv_patch, 0,
                                        1)  # 通过上一轮获得的adv_path得到 ae image (即 applied patch)
            
            with torch.no_grad():  # infer一次，获得victim img 的detect结果 (无须构建计算图)
                output_clean = self.model(victim_img)[0].detach()  # (batch=1, 25200, 85)
            
            # infer一次，获得 ae img 的detect 结果 (因为后面要对 adv patch 求导，因此要构建计算图)
            output_patch = self.model(applied_patch)[0]  # (batch=1, 25200, 85)
            
            # 计算 loss
            loss = self.loss_func(
                loss_hypers,
                target_class,
                (self.patch_size_H, self.patch_size_W),
                self.device,
                self.conf_threshold,
                self.iou_threshold,
                self.patch_conf_threshold,
                output_clean, output_patch
            )
            
            # ---------------------- FGSM ----------------------- #
            self.model.zero_grad()  # ToDo 为什么这里要设置model的grad全为0？它们会累积grad吗？
            data_grad = torch.autograd.grad(loss, adv_patch)[0]  # (3, 640, 640)
            
            # Collect the element wise sign of the data gradient
            # Create the perturbed image by adjusting each pixel of the input image
            tmp_adv_patch = adv_patch - fgsm_epsilon * data_grad.sign()
            
            # Adding clipping to maintain [0,1] range
            tmp_adv_patch = torch.clamp(tmp_adv_patch, 0, 1).detach()
            
            # --------------------- projection --------------------- #
            perturbation = tmp_adv_patch - patch
            norm = torch.sqrt(torch.sum(torch.square(perturbation)))
            factor = min(1, pgd_epsilon / norm.item())  # torch.divide(epsilon, norm.numpy()[0]))
            adv_patch = (torch.clip(patch + perturbation * factor, 0, 1))  # .detach()
            
            # 保存 训练的 loss
            if curr_iter % 20 == 0:
                print('[Log Exe time] {} => {}'.format(curr_iter, time.time() - start))
                save_json_path = trailDir / 'log-zyx-phantom-attack-iter={}-lr={}.log'.format(self.max_iter, fgsm_epsilon)
                with open(str(save_json_path), 'w') as wf:
                    loss_dict = {
                        'max_objects_loss': self.max_objects_loss_list,
                        'pass_to_NMS': self.pass_to_NMS_list,
                        'bboxes_area_loss': self.bboxes_area_loss_list,
                        'benign_detection_loss': self.benign_detection_loss_list,
                        'train_loss': self.train_loss_list,
                        'time': self.time
                    }
                    to_json = json.dumps(loss_dict, indent=4)
                    wf.write(to_json)
                
                save_patch = trailDir / 'save_patch\iter={}-lr={}'.format(self.max_iter, fgsm_epsilon)
                save_patch.mkdir(parents=True, exist_ok=True)
                transforms.ToPILImage()(adv_patch).save(save_patch / 'iter-{}.PNG'.format(curr_iter))


def run_attack(max_iter, fgsm_epsilon):
    phantomGenerator = PhantomGenerator(
        model_name='yolov5',
        model_ckpt_path=str(pretrainedCheckpointDir / 'yolov5s.pt'),
        max_iter=max_iter,
        use_cuda=True,
        conf_threshold=0.25,
        iou_threshold=0.45,
        patch_conf_threshold=0.001
    )
    
    loss_hypers = {
        'lambda_1': 1,
        'lambda_2': 0,
        'lambda_2a': 5,
        'lambda_2b': 0,
        'lambda_3': 0
    }
    
    phantomGenerator.generate_phantom(
        loss_hypers=loss_hypers,
        target_class=0,  # for class = person
        pgd_epsilon=70,  # restrict the adv_patch is no larger than init patch(i.e., all 0) by distance of 70 (0~1 pixel value, in l2 norm)
        fgsm_epsilon=fgsm_epsilon  # for each iter, the perturbation range (+- 0.005) in pixel value (full range: 0~1)
    )


if __name__ == '__main__':
    run_attack(1000, 0.005)
    # run_attack(1000, 0.01)
    # run_attack(1000, 0.05)
    # run_attack(10000, 0.0005)
    # run_attack(10000, 0.0001)

# About the fgsm epsilon
# (0.01 daedalus adam learning rate)
# (0.0005 phantom sponge attack iter epsilon)  -> 每个epoch更新约170次；30 epoch ~ 5k 次
# my trial: 0.0001 / 0.0005 / 0.001 / 0.005 / 0.01 / 0.05

# Choose hyper
# lr = 0.0001 -> iter = 2k
# lr = 0.0005 -> iter = ~700
# lr = 0.005 -> iter = ~250
# lr = 0.01 -> iter =  ~500
# lr = 0.05 -> Nope

# Also, test the victim img with different start up
# since the output_patch = self.model(applied_patch)[0] (applied_patch = original + adv_patch)
# so, higher start up may make the applied_patch to large, and yolov5 model detect nothing

