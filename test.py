import torch
import argparse
import os
from configs.defaults import cfg
from utils.misc import mkdir
from utils.logger import *
from utils.checkpoint import CheckPointer
from models import build_model
from input import get_verse_list
from utils.data import *
from torchvision import transforms, utils
import scipy.misc
from torch.utils.data import DataLoader
from utils.trainer import do_train
import imageio
from utils.metrics import *
import torch.nn.functional as F
import scipy.io

@torch.no_grad()
def pred(cfg):
    device = torch.device(cfg.TEST.DEVICE)
    model = build_model(cfg).to(device)
    lr = cfg.SOLVER.LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    arguments = {"iteration": 0, "epoch": 0}
    checkpointer = CheckPointer(model, optimizer, cfg.OUTPUT_DIR)
    extra_checkpoint_data = checkpointer.load(is_val=True)
    arguments.update(extra_checkpoint_data)

    model.eval()
    is_test =  0
    dataset = ProjectionDataset(cfg=cfg, mat_dir=cfg.MAT_DIR_TEST if is_test else cfg.MAT_DIR_VAL
                                , input_img_dir=cfg.INPUT_IMG_DIR_TEST if is_test else cfg.INPUT_IMG_DIR_VAL,
                                transform=transforms.Compose([ToTensor()]))
    test_loader = DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4)
    mkdir(os.path.join(cfg.OUTPUT_DIR, "jpg_val"))
    #os.system("rm " + os.path.join(cfg.OUTPUT_DIR, "jpg_val/*"))
    name_list = []
    whole_step_list = []
    score_list = []
    position_cor_list = []
    position_sag_list = []

    if is_test == 0:
        val_file_list = glob.glob(cfg.MAT_DIR_VAL + '*.mat')
        val_file_list.sort()

        gt_label_list = []
        for idx in range(len(val_file_list)):
            gt_label_list.append(json.load(
                open(cfg.ORIGINAL_PATH + 'pos/' + val_file_list[idx][len(cfg.MAT_DIR_VAL):-4] + '_ctd.json', "rb")))

    for idx, sample in enumerate(test_loader):
        print(idx)
        input_cor = sample["input_cor"].float().to(device)
        input_sag = sample["input_sag"].float().to(device)
        sag_pad = sample["sag_pad"]
        cor_pad = sample["cor_pad"]
        if is_test == 0:
            gt_cor = sample["gt_cor"].float().to(device)
            gt_sag = sample["gt_sag"].float().to(device)
        output_sag, output_cor = model(input_sag, input_cor)

        for batch_num in range(input_cor.shape[0]):
            output_sag[batch_num, :, :sag_pad[2][batch_num], :] = 0
            output_sag[batch_num, :, :, output_sag.shape[3] - sag_pad[1][batch_num]:] = 0
            output_sag[batch_num, :, output_sag.shape[2] - sag_pad[3][batch_num]:, :] = 0
            output_sag[batch_num, :, :, :sag_pad[0][batch_num]] = 0

            output_cor[batch_num, :, :cor_pad[2][batch_num], :] = 0
            output_cor[batch_num, :, :, output_cor.shape[3] - cor_pad[1][batch_num]:] = 0
            output_cor[batch_num, :, output_cor.shape[2] - cor_pad[3][batch_num]:, :] = 0
            output_cor[batch_num, :, :, :cor_pad[0][batch_num]] = 0

        if is_test:
            for j in range(input_cor.shape[0]):
                imageio.imwrite(cfg.OUTPUT_DIR + "jpg_val/" + sample['name'][j] + "_input_cor.jpg", torch.squeeze(input_cor[j, :, :]).cpu().detach().numpy())
                imageio.imwrite(cfg.OUTPUT_DIR + "jpg_val/" + sample['name'][j] + "_input_sag.jpg", torch.squeeze(input_sag[j, :, :]).cpu().detach().numpy())
                imageio.imwrite(cfg.OUTPUT_DIR + "jpg_val/" + sample['name'][j] + "_output_cor.jpg", 30 * np.max(torch.squeeze(output_cor[j, 1:25, :, :]).cpu().detach().numpy(), axis=0))
                imageio.imwrite(cfg.OUTPUT_DIR + "jpg_val/" + sample['name'][j] + "_output_sag.jpg", 30 * np.max(torch.squeeze(output_sag[j, 1:25, :, :]).cpu().detach().numpy(), axis=0))


        #for c_num in range(24):
            #output_sag[:, c_num + 1, :, :] = output_sag[:, c_num+1, :, :] * (output_sag[:, 0, :, :].max() - output_sag[:, 0, :, :])
            #output_cor[:, c_num + 1, :, :] = output_cor[:, c_num + 1, :, :] * (output_cor[:, 0, :, :].max() - output_cor[:, 0, :, :])
        position, position_batch_cor , position_batch_sag = pred_pos_3(device, output_sag[:, 1:25, :, :], output_cor[:, 1:25, :, :], sample['direction'],
                            sample['crop_info'], sample['spacing'], sample['cor_pad'], sample['sag_pad'])
        # position = pred_pos(device, output_sag[:, 1:25, :, :], output_cor[:, 1:25, :, :], sample['direction'],
        #                       sample['crop_info'], sample['spacing'], sample['cor_pad'], sample['sag_pad'])

        if idx == 0:
            for step in range(position.shape[0]):
                whole_step_list.append([])
                score_list.append([])

        for j in range(input_sag.shape[0]):
            position_cor_list.append(position_batch_cor[j, :, :])
            position_sag_list.append(position_batch_sag[j, :, :])

        for j in range(input_sag.shape[0]):
            for step in range(position.shape[0]):
                whole_step_list[step].append(
                    create_centroid_pos([sample['direction_sitk'][0][j], sample['direction_sitk'][1][j], sample['direction_sitk'][2][j],
                                         sample['direction_sitk'][3][j], sample['direction_sitk'][4][j], sample['direction_sitk'][5][j],
                                         sample['direction_sitk'][6][j], sample['direction_sitk'][7][j], sample['direction_sitk'][8][j]],
                                    [sample['spacing'][0][j], sample['spacing'][1][j], sample['spacing'][2][j]],
                                    [sample['size_raw'][0][j], sample['size_raw'][1][j], sample['size_raw'][2][j]],
                                    position[step, j, :, 0:3])
                )
                score_list[step].append(position[step, j, :, 3])
            name_list.append(sample["name"][j])

    id_rate = list(range(position.shape[0]))
    id_rate_gt = list(range(position.shape[0]))
    if is_test == 0:
        for step in range(position.shape[0]):
            id_rate[step], id_rate_gt[step] = Get_Identification_Rate(gt_label_list, whole_step_list[step])

    if is_test:
        torch.save({"pred_list": whole_step_list[0], 'score': score_list[0],
                   'pred_cor_list': position_cor_list, 'pred_sag_list':position_sag_list,
                   'name':name_list}, "pred_list/pred_test.pth")
    else:
        print("id_rate: ", id_rate)
        print("id_rate_gt: ", id_rate_gt)
        torch.save({"pred_list": whole_step_list[0], 'score': score_list[0], 'name': name_list, 'gt_list': gt_label_list,
                    'pred_cor_list': position_cor_list, 'pred_sag_list':position_sag_list},
                    "pred_list/pred.pth")




def main():
    torch.cuda.empty_cache()
    # some configs, including yaml file
    parser = argparse.ArgumentParser(description='Btrfly Net Training with Pytorch')
    parser.add_argument(
        "--config_file",
        default="configs/btrfly.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--log_step", default=1, type=int, help="print logs every log_step")
    parser.add_argument("--save_step", default=50, type=int, help="save checkpoint every save_step")
    parser.add_argument("--eval_step", default=10, type=int, help="evaluate dataset every eval_step, disabled if eval_step <= 0")
    parser.add_argument("--use_tensorboard", default=1, type=int, help="use visdom to illustrate training process, unless use_visdom == 0")
    args = parser.parse_args()

    # enable inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware
    # so it helps increase training speed
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # use YACS as the config manager, see https://github.com/rbgirshick/yacs for more info
    # cfg contains all the configs set by configs/defaults and overrided by config_file (see line 13)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    # make output directory designated by OUTPUT_DIR if necessary
    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    # set up 2 loggers
    # logger_all can print time and logger's name
    # logger_message only print message
    # it will print info to stdout and to OUTPUT_DIR/log.txt (way: append)
    logger_all = setup_colorful_logger(
        "main",
        save_dir=os.path.join(cfg.OUTPUT_DIR, 'log.txt'),
        format="include_other_info")
    logger_message = setup_colorful_logger(
        "main_message",
        save_dir=os.path.join(cfg.OUTPUT_DIR, 'log.txt'),
        format="only_message")

    # print config info (cfg and args)
    # args are obtained by command line
    # cfg is obtained by yaml file and defaults.py in configs/
    separator(logger_message)
    logger_message.warning(" ---------------------------------------")
    logger_message.warning("|              Your config:             |")
    logger_message.warning(" ---------------------------------------")
    logger_message.info(args)
    logger_message.warning(" ---------------------------------------")
    logger_message.warning("|      Running with entire config:      |")
    logger_message.warning(" ---------------------------------------")
    logger_message.info(cfg)
    separator(logger_message)

    pred(cfg)

if __name__ == '__main__':
    main()