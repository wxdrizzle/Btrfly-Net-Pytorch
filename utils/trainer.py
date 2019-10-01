import collections
import datetime
import os
import time
from utils.inference import *
import glob
import torch
import torch.nn as nn
import torch.nn.functional as func
from scipy.io import loadmat
from utils.metrics import *
import cv2


def compute_loss(gt_sag, gt_cor, output_sag, output_cor, w_front, w_side, device, sag_pad, cor_pad):
    # gt_sag_segment = torch.FloatTensor(gt_sag.size()).to(device)
    # gt_cor_segment = torch.FloatTensor(gt_cor.size()).to(device)
    #
    # gt_sag_segment[:, 1:25, :, :] = torch.where(gt_sag[:, 1:25, :, :] > 0.6,
    #                                             torch.full_like(gt_sag[:, 1:25, :, :], 1),
    #                                             torch.full_like(gt_sag[:, 1:25, :, :], 0))
    # gt_sag_segment[:, 0, :, :] = torch.where(gt_sag[:, 0, :, :] <= 0.4,
    #                                             torch.full_like(gt_sag[:, 0, :, :], 1),
    #                                             torch.full_like(gt_sag[:, 0, :, :], 0))
    # gt_cor_segment[:, 1:25, :, :] = torch.where(gt_cor[:, 1:25, :, :] > 0.6,
    #                                             torch.full_like(gt_cor[:, 1:25, :, :], 1),
    #                                             torch.full_like(gt_cor[:, 1:25, :, :], 0))
    # gt_cor_segment[:, 0, :, :] = torch.where(gt_cor[:, 0, :, :] <= 0.4,
    #                                          torch.full_like(gt_cor[:, 0, :, :], 1),
    #                                          torch.full_like(gt_cor[:, 0, :, :], 0))

    loss_MSE_sag = torch.sum(torch.pow((gt_sag - output_sag), 2))
    loss_MSE_cor = torch.sum(torch.pow((gt_cor - output_cor), 2))

    product_sag = -func.log_softmax(output_sag, dim=1) * func.softmax(gt_sag, dim=1)
    product_cor = -func.log_softmax(output_cor, dim=1) * func.softmax(gt_cor, dim=1)
    for batch_num in range(gt_cor.shape[0]):
        product_sag[batch_num, :, :sag_pad[2][batch_num], :] = 0
        product_sag[batch_num, :, :, product_sag.shape[3] - sag_pad[1][batch_num]:] = 0
        product_sag[batch_num, :, product_sag.shape[2] - sag_pad[3][batch_num]:, :] = 0
        product_sag[batch_num, :, :, :sag_pad[0][batch_num]] = 0

        product_cor[batch_num, :, :cor_pad[2][batch_num], :] = 0
        product_cor[batch_num, :, :, product_cor.shape[3] - cor_pad[1][batch_num]:] = 0
        product_cor[batch_num, :, product_cor.shape[2] - cor_pad[3][batch_num]:, :] = 0
        product_cor[batch_num, :, :, :cor_pad[0][batch_num]] = 0



    loss_cross_entropy_sag = torch.sum(torch.sum(torch.sum(torch.sum(product_sag, dim=2), dim=2), dim=0) * w_side)
    loss_cross_entropy_cor = torch.sum(torch.sum(torch.sum(torch.sum(product_cor, dim=2), dim=2), dim=0) * w_front)

    return loss_MSE_sag + loss_MSE_cor + loss_cross_entropy_cor + loss_cross_entropy_sag



def do_train(cfg, args, model, model_D1, model_D2, data_loader, optimizer, optimizer_D1, optimizer_D2, checkpointer, device, arguments):
    #
    logger = setup_colorful_logger("trainer", save_dir=os.path.join(cfg.OUTPUT_DIR, 'log.txt'), format="include_other_info")
    logger.warning("Start training ...")
    logger_val = setup_colorful_logger("evaluator", save_dir=os.path.join(cfg.OUTPUT_DIR, 'log.txt'), format="include_other_info")
    w = loadmat(cfg.TRAIN_WEIGHT)
    w_front, w_side = torch.Tensor(w["front"]).to(device), torch.Tensor(w["side"]).to(device)

    model.train()
    if None not in (model_D1, model_D2, optimizer_D1, optimizer_D2):
        m = torch.tensor(32).to(device)
        model_D1.train()
        model_D2.train()
    if args.use_tensorboard:
        import tensorboardX
        summary_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = cfg.SOLVER.MAX_ITER
    iteration = arguments["iteration"]
    start_epoch = arguments["epoch"]
    list_loss_val = arguments["list_loss_val"]

    start_training_time = time.time()
    for epoch in range(round(max_iter/len(data_loader)))[start_epoch+1:]:
        arguments["epoch"] = epoch
        loss_show = 0
        if None not in (model_D1, model_D2, optimizer_D1, optimizer_D2):
            loss_show_D1 = 0
            loss_show_D2 = 0
        ins_num = 0
        for idx, sample in enumerate(data_loader):
            iteration = iteration + 1
            arguments["iteration"] = iteration

            input_cor_padded = sample["input_cor"].float().to(device)
            input_sag_padded = sample["input_sag"].float().to(device)
            gt_cor = sample["gt_cor"].float().to(device)
            gt_sag = sample["gt_sag"].float().to(device)
            cor_pad = sample["cor_pad"]
            sag_pad = sample["sag_pad"]

            output_sag, output_cor = model(input_sag_padded, input_cor_padded)

            for batch_num in range(gt_cor.shape[0]):
                output_sag[batch_num, :, :sag_pad[2][batch_num], :] = 0
                output_sag[batch_num, :, :, output_sag.shape[3] - sag_pad[1][batch_num]:] = 0
                output_sag[batch_num, :, output_sag.shape[2] - sag_pad[3][batch_num]:, :] = 0
                output_sag[batch_num, :, :, :sag_pad[0][batch_num]] = 0

                output_cor[batch_num, :, :cor_pad[2][batch_num], :] = 0
                output_cor[batch_num, :, :, output_cor.shape[3] - cor_pad[1][batch_num]:] = 0
                output_cor[batch_num, :, output_cor.shape[2] - cor_pad[3][batch_num]:, :] = 0
                output_cor[batch_num, :, :, :cor_pad[0][batch_num]] = 0

            if None not in (model_D1, model_D2, optimizer_D1, optimizer_D2):
                output_fake_D1 = model_D1(output_sag.detach())
                output_fake_D2 = model_D2(output_cor.detach())
                output_gt_D1 = model_D1(gt_sag.detach())
                output_gt_D2 = model_D2(gt_cor.detach())
                loss_D1 = output_gt_D1 + torch.max(torch.tensor(0).float().to(device), m - output_fake_D1)
                loss_D2 = output_gt_D2 + torch.max(torch.tensor(0).float().to(device), m - output_fake_D2)

                loss_show_D1 += loss_D1.item()
                loss_show_D2 += loss_D2.item()

            ins_num += gt_cor.size(0)

            if None not in (model_D1, model_D2, optimizer_D1, optimizer_D2):
                optimizer_D1.zero_grad()
                loss_D1.backward()
                optimizer_D1.step()

                optimizer_D2.zero_grad()
                loss_D2.backward()
                optimizer_D2.step()


            loss_G = compute_loss(gt_sag, gt_cor, output_sag, output_cor, w_front, w_side, device, sag_pad, cor_pad)
            if None not in (model_D1, model_D2, optimizer_D1, optimizer_D2):
                loss_G = loss_G + model_D1(output_sag) + model_D2(output_cor)
            loss_show += loss_G.item()
            optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()


        if ins_num != len(glob.glob(pathname=cfg.MAT_DIR_TRAIN + "*.mat")):
            raise Exception("Instance number is not equal to sum of batch sizes!")

        if epoch % args.log_step == 0:
            if None in (model_D1, model_D2, optimizer_D1, optimizer_D2):
                logger.info("epoch: {epoch:05d}, iter: {iter:06d}, loss_G: {loss_G}"
                            .format(epoch=epoch, iter=iteration, loss_G=loss_show/ins_num))
            else:
                logger.info("epoch: {epoch:05d}, iter: {iter:06d}, loss_G: {loss_G}, loss_D1: {loss_D1}, loss_D2: {loss_D2}"
                            .format(epoch=epoch, iter=iteration, loss_G=loss_show/ins_num, loss_D1=loss_show_D1/ins_num, loss_D2=loss_show_D2/ins_num))
            if summary_writer:
                summary_writer.add_scalar('loss_G', loss_show/ins_num, global_step=iteration)
                if None not in (model_D1, model_D2, optimizer_D1, optimizer_D2):
                    summary_writer.add_scalar('loss_D1', loss_show_D1 / ins_num, global_step=iteration)
                    summary_writer.add_scalar('loss_D2', loss_show_D2 / ins_num, global_step=iteration)


        if args.eval_step > 0 and epoch % args.eval_step == 0 and not iteration == max_iter:
            loss_val, id_rate, id_rate_gt = do_evaluation(cfg, model, summary_writer, iteration)
            logger_val.error("epoch: {epoch:05d}, iter: {iter:06d}, evaluation_loss: {loss}, \nid_rate: {id_rate}, \nid_rate_gt: {id_rate_gt}, "
                             .format(epoch=epoch, iter=iteration, loss=loss_val, id_rate=id_rate, id_rate_gt=id_rate_gt))
            best_id_rate_gt = - max(id_rate_gt)
            max_loss_iter = max(list_loss_val, key=list_loss_val.get) if len(list_loss_val) else 999
            min_loss_iter = min(list_loss_val, key=list_loss_val.get) if len(list_loss_val) else -1
            if len(list_loss_val) == 0:
                logger_val.warning("Have no saved model, saving first model_{:06d}. ".format(iteration))
                checkpointer.save("model_{:06d}".format(iteration), is_last=False, is_best=True, **arguments)
                list_loss_val[str(iteration)] = best_id_rate_gt
            elif len(list_loss_val) < cfg.SOLVER.SAVE_NUM:
                if list_loss_val[min_loss_iter] > best_id_rate_gt:
                    logger_val.warning("Have saved {:02d} models, "
                                       "saving newest (best) model_{:06d}. ".format(len(list_loss_val), iteration))
                    checkpointer.save("model_{:06d}".format(iteration), is_last=False, is_best=True, **arguments)
                else:
                    logger_val.warning("Have saved {:02d} models, "
                                       "saving newest (NOT best) model_{:06d}. ".format(len(list_loss_val), iteration))
                    checkpointer.save("model_{:06d}".format(iteration), is_last=False, is_best=False, **arguments)
                list_loss_val[str(iteration)] = best_id_rate_gt
            else:
                if list_loss_val[max_loss_iter] >= best_id_rate_gt:
                    if list_loss_val[min_loss_iter] > best_id_rate_gt:
                        logger_val.warning("Have saved {:02d} models, "
                                           "deleting the worst saved model_{:06d} and "
                                           "saving newest (best) model_{:06d}. ".format(cfg.SOLVER.SAVE_NUM, int(max_loss_iter), iteration))
                        checkpointer.save("model_{:06d}".format(iteration), is_last = False, is_best=True, **arguments)
                    else:
                        logger_val.warning("Have saved {:02d} models, "
                                           "deleting the worst saved model_{:06d} and "
                                           "saving newest (NOT best) model_{:06d}. ".format(cfg.SOLVER.SAVE_NUM, int(max_loss_iter), iteration))
                        checkpointer.save("model_{:06d}".format(iteration), is_last=False, is_best=False, **arguments)
                    del list_loss_val[max_loss_iter]
                    os.system("rm " + cfg.OUTPUT_DIR + "model_{:06d}.pth".format(int(max_loss_iter)))
                    list_loss_val[str(iteration)] = best_id_rate_gt
                else:
                    logger_val.warning("Have saved {:02d} models, "
                                       "newest model_{:06d} is the worst. "
                                       "No model is saved or deleted in the best-model list. ".format(cfg.SOLVER.SAVE_NUM, iteration))
            os.system("rm " + cfg.OUTPUT_DIR + "model_last.pth")
            checkpointer.save("model_last", is_last=True, is_best=False, **arguments)

            if summary_writer:
                summary_writer.add_scalar('val_loss', loss_val, global_step=iteration)
            model.train()

        if iteration > max_iter:
            break

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.warning("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model