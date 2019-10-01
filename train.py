import torch
import argparse
from configs.defaults import cfg
from utils.misc import mkdir
from utils.logger import *
from utils.checkpoint import CheckPointer
from models import build_model
from utils.data import *
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.trainer import do_train

def train(cfg, args):
    # set default device
    device = torch.device(cfg.MODEL.DEVICE)
    # build Butterfly Net as [model]
    model = build_model(cfg, name="Btrfly").to(device)

    #build discriminator nets as [model_D1] and [model_D2] if necessary
    model_D1, model_D2 = None, None
    if cfg.MODEL.USE_GAN:
        model_D1 = build_model(cfg, name="EBGAN").to(device)
        model_D2 = build_model(cfg, name="EBGAN").to(device)
        print(model_D1)

    #if you need to visualize the Net, uncomment these codes
    """
    input1 = torch.rand(3, 1, 128, 128)  
    input2 = torch.rand(3, 1, 128, 128)
    with SummaryWriter(comment='BtrflyNet') as w:
        w.add_graph(model, (input1, input2, ))
    """

    # learning rate
    lr = cfg.SOLVER.LR
    # optimizer of [model]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    #optimizers of [model_D1] and [model D2] if necessary
    optimizer_D1, optimizer_D2 = None, None
    if cfg.MODEL.USE_GAN:
        optimizer_D1 = torch.optim.Adam(model_D1.parameters(), lr=lr)
        optimizer_D2 = torch.optim.Adam(model_D2.parameters(), lr=lr)

    # update [checkpointer] if necessary
    # except iteration and epoch numbers,
    # [arguments] also has a list which contains the information of the best several models,
    # including their numbers and their validation losses
    arguments = {"iteration": 0, "epoch": 0, "list_loss_val": {}}
    checkpointer = CheckPointer(model, optimizer, cfg.OUTPUT_DIR)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    # build training set from the directory designated by cfg
    dataset = ProjectionDataset(cfg=cfg,
                                mat_dir=cfg.MAT_DIR_TRAIN,
                                input_img_dir=cfg.INPUT_IMG_DIR_TRAIN,
                                transform=transforms.Compose([ToTensor()]),
                                )
    train_loader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=4)


    return do_train(cfg, args, model, model_D1, model_D2, train_loader, optimizer, optimizer_D1, optimizer_D2, checkpointer, device, arguments)




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
    parser.add_argument("--save_step", default=5, type=int, help="save checkpoint every save_step")
    parser.add_argument("--eval_step", default=5, type=int, help="evaluate dataset every eval_step, disabled if eval_step <= 0")
    parser.add_argument("--use_tensorboard", default=1, type=int, help="use visdom to illustrate training process, unless use_visdom == 0")
    parser.add_argument("--train_from_no_checkpoint", default=1, type=int, help="train_from_no_checkpoint")
    args = parser.parse_args()

    # enable inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware
    # so it helps increase training speed
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # use YACS as the config manager, see https://github.com/rbgirshick/yacs for more info
    # cfg contains all the configs set by configs/defaults and overrided by config_file
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    # make output directory designated by OUTPUT_DIR if necessary
    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)
    # if you need, this removes the results related to last training
    if args.train_from_no_checkpoint:
        os.system("rm -r " + os.path.join(cfg.OUTPUT_DIR, "*"))

    # logger_message help print message
    # it will also print info to stdout and to OUTPUT_DIR/log.txt (way: append)
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

    train(cfg, args)

if __name__ == '__main__':
    main()