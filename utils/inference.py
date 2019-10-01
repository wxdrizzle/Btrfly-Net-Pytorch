import torch
import argparse
import os
from configs.defaults import cfg
from utils.misc import mkdir
from utils.logger import *
from utils.checkpoint import CheckPointer
from models import build_model
from input import get_verse_list
from utils import trainer
from utils.metrics import *
from utils.data import *
from torchvision import transforms, utils
import scipy.misc
from torch.utils.data import DataLoader
import glob
import torch.nn.functional as F

@torch.no_grad()
def do_evaluation(cfg, model, summary_writer, global_step):
    device = torch.device(cfg.MODEL.DEVICE)
    model.eval()

    w = loadmat(cfg.TRAIN_WEIGHT)
    w_front, w_side = torch.Tensor(w["front"]).to(device), torch.Tensor(w["side"]).to(device)

    dataset = ProjectionDataset(cfg=cfg, mat_dir=cfg.MAT_DIR_VAL, input_img_dir=cfg.INPUT_IMG_DIR_VAL,
                                transform=transforms.Compose([ToTensor()]))
    val_loader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4)

    val_loss = 0
    val_num = 0
    whole_step_list = []
    whole_step_list_softmax = []
    whole_step_list_norm = []
    val_file_list = glob.glob(cfg.MAT_DIR_VAL + '*.mat')
    val_file_list.sort()

    gt_label_list = []
    for idx in range(len(val_file_list)):
        gt_label_list.append(json.load(open(cfg.ORIGINAL_PATH + 'pos/' + val_file_list[idx][len(cfg.MAT_DIR_VAL):-4] + '_ctd.json', "rb")))

    for idx, sample in enumerate(val_loader):
        input_cor = sample["input_cor"].float().to(device)
        input_sag = sample["input_sag"].float().to(device)
        gt_cor = sample["gt_cor"].float().to(device)
        gt_sag = sample["gt_sag"].float().to(device)
        cor_pad = sample["cor_pad"]
        sag_pad = sample["sag_pad"]

        output_sag, output_cor = model(input_sag, input_cor)

        for batch_num in range(gt_cor.shape[0]):
            output_sag[batch_num, :, :sag_pad[2][batch_num], :] = 0
            output_sag[batch_num, :, :, output_sag.shape[3] - sag_pad[1][batch_num]:] = 0
            output_sag[batch_num, :, output_sag.shape[2] - sag_pad[3][batch_num]:, :] = 0
            output_sag[batch_num, :, :, :sag_pad[0][batch_num]] = 0

            output_cor[batch_num, :, :cor_pad[2][batch_num], :] = 0
            output_cor[batch_num, :, :, output_cor.shape[3] - cor_pad[1][batch_num]:] = 0
            output_cor[batch_num, :, output_cor.shape[2] - cor_pad[3][batch_num]:, :] = 0
            output_cor[batch_num, :, :, :cor_pad[0][batch_num]] = 0

        for i in range(output_sag.shape[0]):
            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "side_input",
                                     (input_sag[i, :, :, :]-torch.max(input_sag[i, :, :, :]))/(torch.max(input_sag[i, :, :, :])-torch.min(input_sag[i, :, :, :])),
                                     global_step=global_step)
            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "front_input",
                                     (input_cor[i, :, :, :]-torch.max(input_cor[i, :, :, :]))/(torch.max(input_cor[i, :, :, :])-torch.min(input_cor[i, :, :, :])),
                                     global_step=global_step)
            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "side_output",
                                     torch.max(output_sag[i, 1:25, :, :], dim=0)[0].view(1, output_sag.shape[2], output_sag.shape[3]),
                                     global_step=global_step)
            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "front_output",
                                     torch.max(output_cor[i, 1:25, :, :], dim=0)[0].view(1, output_cor.shape[2], output_cor.shape[3]),
                                     global_step=global_step)
            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "side_output_bkgd",
                                     output_sag[i, 0, :, :].view(1, output_sag.shape[2], output_sag.shape[3]),
                                     global_step=global_step)
            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "front_output_bkgd",
                                     output_cor[i, 0, :, :].view(1, output_cor.shape[2],output_cor.shape[3]),
                                     global_step=global_step)

            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "side_gt",
                                     torch.max(gt_sag[i, 1:25, :, :], dim=0)[0].view(1, gt_sag.shape[2], gt_sag.shape[3]),
                                     global_step=global_step)
            summary_writer.add_image(str(idx) + "_" + str(i) + "_" + "front_gt",
                                     torch.max(gt_cor[i, 1:25, :, :], dim=0)[0].view(1, gt_cor.shape[2], gt_cor.shape[3]),
                                     global_step=global_step)


        position = pred_pos_3(device, output_sag[:, 1:25, :, :], output_cor[:, 1:25, :, :], sample['direction'], sample['crop_info'], sample['spacing'], sample['cor_pad'], sample['sag_pad'])[0]

        if idx == 0:
            for step in range(position.shape[0]):
                whole_step_list.append([])

        for j in range(input_sag.shape[0]):
            for step in range(position.shape[0]):
                whole_step_list[step].append(
                    create_centroid_pos([sample['direction_sitk'][0][j], sample['direction_sitk'][1][j], sample['direction_sitk'][2][j],
                                         sample['direction_sitk'][3][j], sample['direction_sitk'][4][j], sample['direction_sitk'][5][j],
                                         sample['direction_sitk'][6][j], sample['direction_sitk'][7][j], sample['direction_sitk'][8][j]],
                                    [sample['spacing'][0][j], sample['spacing'][1][j], sample['spacing'][2][j]],
                                    [sample['size_raw'][0][j], sample['size_raw'][1][j], sample['size_raw'][2][j]],
                                    position[step, j, :, :])
                )

        val_loss = val_loss + trainer.compute_loss(gt_sag[:, :, :, :], gt_cor[:, :, :, :], output_sag, output_cor, w_front, w_side, device, sag_pad, cor_pad)
        val_num += gt_cor.size(0)



    id_rate = list(range(position.shape[0]))
    id_rate_gt = list(range(position.shape[0]))
    for step in range(position.shape[0]):
        id_rate[step], id_rate_gt[step] = Get_Identification_Rate(gt_label_list, whole_step_list[step])

    summary_writer.add_scalar('id_rate_val', max(id_rate), global_step=global_step)
    summary_writer.add_scalar('id_rate_val_gt', max(id_rate_gt), global_step=global_step)

    if val_num != len(glob.glob(pathname=cfg.MAT_DIR_VAL + "*.mat")):
        raise Exception("Validation number is not equal to sum of batch sizes!")

    return val_loss.item() / val_num, id_rate, id_rate_gt