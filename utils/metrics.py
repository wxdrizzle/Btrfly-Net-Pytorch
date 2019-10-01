import torch
import numpy as np
import json
import cv2

Dic = {0:'Z',1:'Y',2:'X'}

@torch.no_grad()
def decom(whole_dict):
    label = []
    dim_X = []
    dim_Y = []
    dim_Z = []
    for index in range(len(whole_dict)):
        current = whole_dict[index]
        label.append(current['label'])
        dim_X.append(current['X'])
        dim_Y.append(current['Y'])
        dim_Z.append(current['Z'])
    return label, dim_X, dim_Y, dim_Z

@torch.no_grad()
def create_centroid_pos(Direction, Spacing, Size, position):
    # dim0, dim1,dim2, label):
    """

    :param Direction,Spacing, Size: from sitk  raw.GetDirection(),GetSpacing(),GetSize()
    :param position:[24,3]
    :return:
    """
    direction = np.round(list(Direction))
    direc0 = direction[0:7:3]
    direc1 = direction[1:8:3]
    direc2 = direction[2:9:3]
    dim0char = Dic[(np.argwhere((np.abs(direc0)) == 1))[0][0]]
    dim1char = Dic[(np.argwhere((np.abs(direc1)) == 1))[0][0]]
    dim2char = Dic[(np.argwhere((np.abs(direc2)) == 1))[0][0]]
    resolution = Spacing
    w, h, c = Size[0], Size[1], Size[2]
    jsonlist = []
    for i in range(24):
        dim0, dim1, dim2 = position[i:i + 1, 0], position[i:i + 1, 1], position[i:i + 1, 2]
        if dim0 >= 0:
            label = i + 1
            if np.sum(direc0) == -1:
                if dim0char == 'X':
                    Jsondim0 = dim0 * resolution[0]
                else:
                    Jsondim0 = (w - dim0) * resolution[0]
            else:
                if dim0char == 'X':
                    Jsondim0 = (w - dim0) * resolution[0]
                else:
                    Jsondim0 = dim0 * resolution[0]

            if np.sum(direc1) == -1:
                if dim1char == 'X':
                    Jsondim1 = dim1 * resolution[1]
                else:
                    Jsondim1 = (h - dim1) * resolution[1]
            else:
                if dim1char == 'X':
                    Jsondim1 = (h - dim1) * resolution[1]
                else:
                    Jsondim1 = dim1 * resolution[1]

            if np.sum(direc2) == -1:
                if dim2char == 'X':
                    Jsondim2 = dim2 * resolution[2]
                else:
                    Jsondim2 = (c - dim2) * resolution[2]
            else:
                if dim2char == 'X':
                    Jsondim2 = (c - dim2) * resolution[2]
                else:
                    Jsondim2 = dim2 * resolution[2]
            jsonlist.append({dim0char: Jsondim0, dim1char: Jsondim1, dim2char: Jsondim2, 'label': label})

    return jsonlist

@torch.no_grad()
def Get_Identification_Rate(ground_truth_list: object, pred_list: object) -> object:
    """

    :param ground_truth_list: dict-->{'X':XXX, 'Y':XXX, 'Z':XXX}
    :param pred_list:
    :return:
    """
    correctpred = 0
    whole_number = 0
    whole_number_gt = 0
    for i in range(len(pred_list)):
        label_GT, dim_X_GT, dim_Y_GT, dim_Z_GT = decom(ground_truth_list[i])
        label_PRED, dim_X_PRED, dim_Y_PRED, dim_Z_PRED = decom(pred_list[i])
        whole_number += len(label_PRED)
        whole_number_gt += len(label_GT)
        for idx in range(len(label_PRED)):
            label_c = label_PRED[idx]
            if label_c in label_GT:
                pos = label_GT.index(label_c)
                dif_X = dim_X_GT[pos] - dim_X_PRED[idx]
                dif_Y = dim_Y_GT[pos] - dim_Y_PRED[idx]
                dif_Z = dim_Z_GT[pos] - dim_Z_PRED[idx]
                distance = pow((pow(dif_X, 2) + pow(dif_Y, 2) + pow(dif_Z, 2)), 0.5)
                if distance < 20:
                    correctpred += 1

    iden_rate = correctpred / whole_number  if whole_number != 0 else 0

    return iden_rate, correctpred / whole_number_gt

@torch.no_grad()
def Get_Localisation_distance(ground_truth, pred):
    """

    :param ground_truth: from each subject
    :param pred:
    :return:
    """
    hit = 0
    distance = 0
    label_GT, dim_X_GT, dim_Y_GT, dim_Z_GT = decom(ground_truth)
    label_PRED, dim_X_PRED, dim_Y_PRED, dim_Z_PRED = decom(pred)
    for idx in range(len(label_PRED)):
        label_c = label_PRED[idx]
        if label_c in label_GT:
            hit += 1
            pos = label_GT.index(label_c)
            dif_X = dim_X_GT[pos] - dim_X_PRED[idx]
            dif_Y = dim_Y_GT[pos] - dim_Y_PRED[idx]
            dif_Z = dim_Z_GT[pos] - dim_Z_PRED[idx]
            distance += pow((pow(dif_X, 2) + pow(dif_Y, 2) + pow(dif_Y, 2)), 0.5)
    if hit == 0 :
        print('ALL MISSED')
        loc_dis = []
    else:
        loc_dis = distance / hit

    return loc_dis

@torch.no_grad()
def Get_Recall_AND_Precision(ground_truth, pred):
    """

    :param ground_truth: from each subject
    :param pred:
    :return:
    """
    hit = 0
    label_GT, dim_X_GT, dim_Y_GT, dim_Z_GT = decom(ground_truth)
    label_PRED, dim_X_PRED, dim_Y_PRED, dim_Z_PRED = decom(pred)
    GT_length = len(label_GT)
    PRED_length = len(label_PRED)
    for idx in range(PRED_length):
        label_c = label_PRED[idx]
        if label_c in label_GT:
            pos = label_GT.index(label_c)
            dif_X = dim_X_GT[pos] - dim_X_PRED[idx]
            dif_Y = dim_Y_GT[pos] - dim_Y_PRED[idx]
            dif_Z = dim_Z_GT[pos] - dim_Z_PRED[idx]
            distance = pow((pow(dif_X, 2) + pow(dif_Y, 2) + pow(dif_Z, 2)), 0.5)
            if distance < 20:
                hit += 1
    Recall = hit / GT_length
    Precision = hit / PRED_length

    return Recall, Precision

@torch.no_grad()
def pred_pos(device, output_sag_batch, output_cor_batch, direction, crop_info, spacing, cor_pad, sag_pad):
    """
    Compute the tensor product between output_sag and output_cor,
    then use argmax to find the position of the ith vertebra in channel i.
    Let's say the original 3D data has a shape of (B, C, d0, d1, d2), normally with C=25.
    Parameters:
        output_sag & output_cor: output of the Btrfly Net
        direc: it should be ('Z', 'Y', 'X') or ('Y', 'X', 'Z'), indicating the direction of the subject
    Return:
        a (BxCx3) tensor about the positions of the bones of every subjects in the batch
    """
    if output_sag_batch.shape[:2] != output_cor_batch.shape[:2]:
        raise Exception("output_sag and output_cor have different batch sizes or channel numbers!")
    B, C = output_sag_batch.shape[0], output_sag_batch.shape[1]
    # threshold to reduce noise
    threshold_noise = 0
    threshold_label = torch.from_numpy(np.arange(0, 0.4, 0.01)).float()
    position_batch = torch.Tensor(len(threshold_label), B, C, 4)
    resolution = 1.0
    ori_d0, ori_d1, ori_d2 = np.zeros(B), np.zeros(B), np.zeros(B)
    for i in range(B):
        direc = (direction[0][i], direction[1][i], direction[2][i])
        if (direc != ('Z', 'Y', 'X')) & (direc != ('Y', 'X', 'Z')):
            raise Exception('Unknown direction!')
        # select ith subject
        output_cor = output_cor_batch[i, :, :, :]
        output_sag = output_sag_batch[i, :, :, :]

        # reduce the noise according to threshold
        reduce_noise_sag = torch.where(output_sag < threshold_noise, torch.full_like(output_sag, 0), output_sag)
        reduce_noise_cor = torch.where(output_cor < threshold_noise, torch.full_like(output_cor, 0), output_cor)
        max_value, max_idx = torch.zeros(24), torch.zeros(24)

        if direc == ('Z', 'Y', 'X'):
            # sag:(C, d1, d2), cor:(C, d0, d2)
            if (output_sag.shape[2] != output_cor.shape[2]):
                raise Exception("sag and cor should have an identical size in the last dimension!")
            d0, d1, d2 = output_cor.shape[1], output_sag.shape[1], output_sag.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - cor_pad[2][i] - cor_pad[3][i], d1 - sag_pad[2][i] - sag_pad[3][i], d2 - cor_pad[0][i] - cor_pad[1][i]
            #extend them to (d0, d1, d2)
            for c_num in range(24):

                reduce_noise_sag_one_cha = reduce_noise_sag[c_num, sag_pad[2][i]:d1-sag_pad[3][i], sag_pad[0][i]:d2-sag_pad[1][i]]
                reduce_noise_cor_one_cha = reduce_noise_cor[c_num, cor_pad[2][i]:d0-cor_pad[3][i], cor_pad[0][i]:d2-cor_pad[1][i]]
                assert reduce_noise_cor_one_cha.shape[1] == reduce_noise_sag_one_cha.shape[1]

                reduce_noise_sag_one_cha = reduce_noise_sag_one_cha.unsqueeze(0).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))
                reduce_noise_cor_one_cha = reduce_noise_cor_one_cha.unsqueeze(1).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))

                product = reduce_noise_cor_one_cha * reduce_noise_sag_one_cha
                # find maximum value for each batch and channel
                max_value[c_num], max_idx[c_num] = torch.max(product.view(-1), dim=0)
        else:
            # sag:(C, d0, d1), cor:(C, d1, d2)
            if (output_sag.shape[2] != output_cor.shape[1]):
                raise Exception("sag and cor should have an identical size in some dimension!")
            d0, d1, d2 = output_sag.shape[1], output_sag.shape[2], output_cor.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - sag_pad[2][i] - sag_pad[3][i], d1 - sag_pad[0][i] - sag_pad[1][i], d2 - cor_pad[0][i] - cor_pad[1][i]
            #extend them to (d0, d1, d2)
            for c_num in range(24):

                reduce_noise_sag_one_cha = reduce_noise_sag[c_num, sag_pad[2][i]:d0 - sag_pad[3][i], sag_pad[0][i]:d1 - sag_pad[1][i]]
                reduce_noise_cor_one_cha = reduce_noise_cor[c_num, cor_pad[2][i]:d1 - cor_pad[3][i], cor_pad[0][i]:d2 - cor_pad[1][i]]
                assert reduce_noise_cor_one_cha.shape[0] == reduce_noise_sag_one_cha.shape[1]

                reduce_noise_sag_one_cha = reduce_noise_sag_one_cha.unsqueeze(2).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))
                reduce_noise_cor_one_cha = reduce_noise_cor_one_cha.unsqueeze(0).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))
                product = reduce_noise_cor_one_cha * reduce_noise_sag_one_cha
                # find maximum value for each batch and channel
                max_value[c_num], max_idx[c_num] = torch.max(product.view(-1), dim=0)

        # translate the indexes to 3D form
        max_idx_x, max_idx_y, max_idx_z = -torch.ones(24), -torch.ones(24), -torch.ones(24)
        for c_num in range(24):
            max_idx_x[c_num], max_idx_y[c_num], max_idx_z[c_num] = \
                max_idx[c_num] // (ori_d1[i] * ori_d2[i]), \
                (max_idx[c_num] % (ori_d1[i] * ori_d2[i])) // ori_d2[i], \
                (max_idx[c_num] % (ori_d1[i] * ori_d2[i])) % ori_d2[i]
        for step in range(len(threshold_label)):
            position_batch[step, i, :, 0] = (max_idx_x.float() * resolution / spacing[0][i] + crop_info['displace'][i, 0, 0])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 1] = (max_idx_y.float() * resolution / spacing[1][i] + crop_info['displace'][i, 0, 1])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 2] = (max_idx_z.float() * resolution / spacing[2][i] + crop_info['displace'][i, 0, 2])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 3] = max_value

    return position_batch


@torch.no_grad()
def pred_pos_2(device, output_sag_batch, output_cor_batch, direction, crop_info, spacing, cor_pad, sag_pad):
    """
    Compute the tensor product between output_sag and output_cor,
    then use argmax to find the position of the ith vertebra in channel i.
    Let's say the original 3D data has a shape of (B, C, d0, d1, d2), normally with C=25.
    Parameters:
        output_sag & output_cor: output of the Btrfly Net
        direc: it should be ('Z', 'Y', 'X') or ('Y', 'X', 'Z'), indicating the direction of the subject
    Return:
        a (BxCx3) tensor about the positions of the bones of every subjects in the batch
    """
    if output_sag_batch.shape[:2] != output_cor_batch.shape[:2]:
        raise Exception("output_sag and output_cor have different batch sizes or channel numbers!")
    B, C = output_sag_batch.shape[0], output_sag_batch.shape[1]
    # threshold to reduce noise
    threshold_noise = 0
    threshold_label = torch.from_numpy(np.arange(0, 0.4, 0.01)).float()
    position_batch = torch.Tensor(len(threshold_label), B, C, 4)
    resolution = 2.0
    ori_d0, ori_d1, ori_d2 = np.zeros(B), np.zeros(B), np.zeros(B)
    for i in range(B):
        direc = (direction[0][i], direction[1][i], direction[2][i])
        if (direc != ('Z', 'Y', 'X')) & (direc != ('Y', 'X', 'Z')):
            raise Exception('Unknown direction!')
        # select ith subject
        output_cor = output_cor_batch[i, :, :, :]
        output_sag = output_sag_batch[i, :, :, :]

        # reduce the noise according to threshold
        reduce_noise_sag = torch.where(output_sag < threshold_noise, torch.full_like(output_sag, 0), output_sag)
        reduce_noise_cor = torch.where(output_cor < threshold_noise, torch.full_like(output_cor, 0), output_cor)
        max_value, max_idx = torch.zeros(24), torch.zeros(24)
        max_idx_x, max_idx_y, max_idx_z = torch.zeros(24), torch.zeros(24), torch.zeros(24)
        if direc == ('Z', 'Y', 'X'):
            # sag:(C, d1, d2), cor:(C, d0, d2)
            if (output_sag.shape[2] != output_cor.shape[2]):
                raise Exception("sag and cor should have an identical size in the last dimension!")
            d0, d1, d2 = output_cor.shape[1], output_sag.shape[1], output_sag.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - cor_pad[2][i] - cor_pad[3][i], d1 - sag_pad[2][i] - sag_pad[3][i], d2 - cor_pad[0][i] - cor_pad[1][i]
            reduce_noise_sag_no_padding = reduce_noise_sag[:, sag_pad[2][i]:d1-sag_pad[3][i], sag_pad[0][i]:d2-sag_pad[1][i]]
            reduce_noise_cor_no_padding = reduce_noise_cor[:, cor_pad[2][i]:d0-cor_pad[3][i], cor_pad[0][i]:d2-cor_pad[1][i]]
            # (24)
            max_value_sag, max_idx_sag = torch.max(reduce_noise_sag_no_padding.contiguous().view(reduce_noise_sag_no_padding.shape[0], -1), dim=1)
            max_value_cor, max_idx_cor = torch.max(reduce_noise_cor_no_padding.contiguous().view(reduce_noise_cor_no_padding.shape[0], -1), dim=1)
            max_idx_sag_x, max_idx_sag_y = max_idx_sag // ori_d2[i], max_idx_sag % ori_d2[i]
            max_idx_cor_x, max_idx_cor_y = max_idx_cor // ori_d2[i], max_idx_cor % ori_d2[i]
            for c_num in range(24):
                if max_value_sag[c_num] > max_value_cor[c_num]:
                    max_idx_x[c_num], max_idx_y[c_num], max_idx_z[c_num] = max_idx_cor_x[c_num], max_idx_sag_x[c_num], max_idx_sag_y[c_num]
                    max_value[c_num] = max_value_sag[c_num]
                else:
                    max_idx_x[c_num], max_idx_y[c_num], max_idx_z[c_num] = max_idx_cor_x[c_num], max_idx_sag_x[c_num], max_idx_cor_y[c_num]
                    max_value[c_num] = max_value_cor[c_num]
        else:
            # sag:(C, d0, d1), cor:(C, d1, d2)
            if (output_sag.shape[2] != output_cor.shape[1]):
                raise Exception("sag and cor should have an identical size in some dimension!")
            d0, d1, d2 = output_sag.shape[1], output_sag.shape[2], output_cor.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - sag_pad[2][i] - sag_pad[3][i], d1 - sag_pad[0][i] - sag_pad[1][i], d2 - cor_pad[0][i] - cor_pad[1][i]
            reduce_noise_sag_no_padding = reduce_noise_sag[:, sag_pad[2][i]:d1 - sag_pad[3][i], sag_pad[0][i]:d2 - sag_pad[1][i]]
            reduce_noise_cor_no_padding = reduce_noise_cor[:, cor_pad[2][i]:d0 - cor_pad[3][i], cor_pad[0][i]:d2 - cor_pad[1][i]]
            # (24)
            max_value_sag, max_idx_sag = torch.max(reduce_noise_sag_no_padding.contiguous().view(reduce_noise_sag_no_padding.shape[0], -1), dim=1)
            max_value_cor, max_idx_cor = torch.max(reduce_noise_cor_no_padding.contiguous().view(reduce_noise_cor_no_padding.shape[0], -1), dim=1)
            max_idx_sag_x, max_idx_sag_y = max_idx_sag // ori_d1[i], max_idx_sag % ori_d1[i]
            max_idx_cor_x, max_idx_cor_y = max_idx_cor // ori_d2[i], max_idx_cor % ori_d2[i]
            for c_num in range(24):
                if max_value_sag[c_num] > max_value_cor[c_num]:
                    max_idx_x[c_num], max_idx_y[c_num], max_idx_z[c_num] = max_idx_sag_x[c_num], max_idx_sag_y[c_num], max_idx_cor_y[c_num]
                    max_value[c_num] = max_value_sag[c_num]
                else:
                    max_idx_x[c_num], max_idx_y[c_num], max_idx_z[c_num] = max_idx_sag_x[c_num], max_idx_cor_x[c_num], max_idx_cor_y[c_num]
                    max_value[c_num] = max_value_cor[c_num]

        for step in range(len(threshold_label)):
            position_batch[step, i, :, 0] = (max_idx_x.float() * resolution / spacing[0][i] + crop_info['displace'][i, 0, 0])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 1] = (max_idx_y.float() * resolution / spacing[1][i] + crop_info['displace'][i, 0, 1])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 2] = (max_idx_z.float() * resolution / spacing[2][i] + crop_info['displace'][i, 0, 2])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 3] = max_value

    return position_batch

@torch.no_grad()
def pred_pos_3(device, output_sag_batch, output_cor_batch, direction, crop_info, spacing, cor_pad, sag_pad):
    """
    Compute the tensor product between output_sag and output_cor,
    then use argmax to find the position of the ith vertebra in channel i.
    Let's say the original 3D data has a shape of (B, C, d0, d1, d2), normally with C=25.
    Parameters:
        output_sag & output_cor: output of the Btrfly Net
        direc: it should be ('Z', 'Y', 'X') or ('Y', 'X', 'Z'), indicating the direction of the subject
    Return:
        a (BxCx3) tensor about the positions of the bones of every subjects in the batch
    """
    if output_sag_batch.shape[:2] != output_cor_batch.shape[:2]:
        raise Exception("output_sag and output_cor have different batch sizes or channel numbers!")
    B, C = output_sag_batch.shape[0], output_sag_batch.shape[1]
    f_size = 7
    # threshold to reduce noise
    threshold_noise = 0
    threshold_label = torch.from_numpy(np.arange(0, 0.2, 0.005)).float()
    position_batch = torch.Tensor(len(threshold_label), B, C, 4)
    position_batch_sag = torch.Tensor(B, C, 3)
    position_batch_cor = torch.Tensor(B, C, 3)
    resolution = 1.0
    ori_d0, ori_d1, ori_d2 = np.zeros(B), np.zeros(B), np.zeros(B)
    for i in range(B):
        direc = (direction[0][i], direction[1][i], direction[2][i])
        if (direc != ('Z', 'Y', 'X')) & (direc != ('Y', 'X', 'Z')):
            raise Exception('Unknown direction!')
        # select ith subject
        output_cor = output_cor_batch[i, :, :, :]
        output_sag = output_sag_batch[i, :, :, :]

        # reduce the noise according to threshold
        reduce_noise_sag = torch.where(output_sag < threshold_noise, torch.full_like(output_sag, 0), output_sag)
        reduce_noise_cor = torch.where(output_cor < threshold_noise, torch.full_like(output_cor, 0), output_cor)
        max_value, max_idx = torch.zeros(24), torch.zeros(24)

        if direc == ('Z', 'Y', 'X'):
            # sag:(C, d1, d2), cor:(C, d0, d2)
            if (output_sag.shape[2] != output_cor.shape[2]):
                raise Exception("sag and cor should have an identical size in the last dimension!")
            d0, d1, d2 = output_cor.shape[1], output_sag.shape[1], output_sag.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - cor_pad[2][i] - cor_pad[3][i], d1 - sag_pad[2][i] - sag_pad[3][i], d2 - cor_pad[0][i] - cor_pad[1][i]


            reduce_noise_sag_no_padding = reduce_noise_sag[:, sag_pad[2][i]:d1 - sag_pad[3][i],
                                          sag_pad[0][i]:d2 - sag_pad[1][i]]
            reduce_noise_cor_no_padding = reduce_noise_cor[:, cor_pad[2][i]:d0 - cor_pad[3][i],
                                          cor_pad[0][i]:d2 - cor_pad[1][i]]

            # for c_num in range(24):
            #
            #     reduce_noise_sag_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_sag_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)
            #     reduce_noise_cor_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_cor_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)

            # (24)
            max_value_sag, max_idx_sag = torch.max(
                reduce_noise_sag_no_padding.contiguous().view(reduce_noise_sag_no_padding.shape[0], -1), dim=1)
            max_value_cor, max_idx_cor = torch.max(
                reduce_noise_cor_no_padding.contiguous().view(reduce_noise_cor_no_padding.shape[0], -1), dim=1)
            max_idx_sag_x, max_idx_sag_y = max_idx_sag // ori_d2[i], max_idx_sag % ori_d2[i]
            max_idx_cor_x, max_idx_cor_y = max_idx_cor // ori_d2[i], max_idx_cor % ori_d2[i]

            #extend them to (d0, d1, d2)
            for c_num in range(24):
                reduce_noise_sag_one_cha = reduce_noise_sag[c_num, sag_pad[2][i]:d1-sag_pad[3][i], sag_pad[0][i]:d2-sag_pad[1][i]]

                reduce_noise_cor_one_cha = reduce_noise_cor[c_num, cor_pad[2][i]:d0-cor_pad[3][i], cor_pad[0][i]:d2-cor_pad[1][i]]
                assert reduce_noise_cor_one_cha.shape[1] == reduce_noise_sag_one_cha.shape[1]

                reduce_noise_sag_one_cha = reduce_noise_sag_one_cha.unsqueeze(0).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))
                reduce_noise_cor_one_cha = reduce_noise_cor_one_cha.unsqueeze(1).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))

                product = reduce_noise_cor_one_cha * reduce_noise_sag_one_cha
                # find maximum value for each batch and channel
                max_value[c_num], max_idx[c_num] = torch.max(product.view(-1), dim=0)
        else:
            # sag:(C, d0, d1), cor:(C, d1, d2)
            if (output_sag.shape[2] != output_cor.shape[1]):
                raise Exception("sag and cor should have an identical size in some dimension!")
            d0, d1, d2 = output_sag.shape[1], output_sag.shape[2], output_cor.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - sag_pad[2][i] - sag_pad[3][i], d1 - sag_pad[0][i] - sag_pad[1][i], d2 - cor_pad[0][i] - cor_pad[1][i]

            reduce_noise_sag_no_padding = reduce_noise_sag[:, sag_pad[2][i]:d1 - sag_pad[3][i],
                                          sag_pad[0][i]:d2 - sag_pad[1][i]]
            reduce_noise_cor_no_padding = reduce_noise_cor[:, cor_pad[2][i]:d0 - cor_pad[3][i],
                                          cor_pad[0][i]:d2 - cor_pad[1][i]]

            # for c_num in range(24):
            #
            #     reduce_noise_sag_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_sag_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)
            #     reduce_noise_cor_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_cor_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)

            # (24)
            max_value_sag, max_idx_sag = torch.max(
                reduce_noise_sag_no_padding.contiguous().view(reduce_noise_sag_no_padding.shape[0], -1), dim=1)
            max_value_cor, max_idx_cor = torch.max(
                reduce_noise_cor_no_padding.contiguous().view(reduce_noise_cor_no_padding.shape[0], -1), dim=1)



            max_idx_sag_x, max_idx_sag_y = max_idx_sag // ori_d1[i], max_idx_sag % ori_d1[i]
            max_idx_cor_x, max_idx_cor_y = max_idx_cor // ori_d2[i], max_idx_cor % ori_d2[i]

            #extend them to (d0, d1, d2)
            for c_num in range(24):

                reduce_noise_sag_one_cha = reduce_noise_sag[c_num, sag_pad[2][i]:d0 - sag_pad[3][i], sag_pad[0][i]:d1 - sag_pad[1][i]]
                reduce_noise_cor_one_cha = reduce_noise_cor[c_num, cor_pad[2][i]:d1 - cor_pad[3][i], cor_pad[0][i]:d2 - cor_pad[1][i]]
                assert reduce_noise_cor_one_cha.shape[0] == reduce_noise_sag_one_cha.shape[1]

                reduce_noise_sag_one_cha = reduce_noise_sag_one_cha.unsqueeze(2).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))
                reduce_noise_cor_one_cha = reduce_noise_cor_one_cha.unsqueeze(0).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))
                product = reduce_noise_cor_one_cha * reduce_noise_sag_one_cha
                # find maximum value for each batch and channel
                max_value[c_num], max_idx[c_num] = torch.max(product.view(-1), dim=0)

        # translate the indexes to 3D form
        max_idx_x, max_idx_y, max_idx_z = -torch.ones(24), -torch.ones(24), -torch.ones(24)
        for c_num in range(24):
            max_idx_x[c_num], max_idx_y[c_num], max_idx_z[c_num] = \
                max_idx[c_num] // (ori_d1[i] * ori_d2[i]), \
                (max_idx[c_num] % (ori_d1[i] * ori_d2[i])) // ori_d2[i], \
                (max_idx[c_num] % (ori_d1[i] * ori_d2[i])) % ori_d2[i]

        if direc == ('Z', 'Y', 'X'):
            for c_num in range(24):
                max_idx_z[c_num] = (max_idx_sag_y[c_num] * max_value_sag[c_num] +  max_idx_cor_y[c_num] * max_value_cor[c_num]) / \
                                   (max_value_sag[c_num]+max_value_cor[c_num])
                # if max_value_sag[c_num] > max_value_cor[c_num]:
                #     max_idx_z[c_num] = max_idx_sag_y[c_num]
                # else:
                #     max_idx_z[c_num] = max_idx_cor_y[c_num]

        else:
            for c_num in range(24):
                max_idx_y[c_num] = (max_idx_sag_y[c_num] * max_value_sag[c_num] + max_idx_cor_x[c_num] * max_value_cor[c_num]) / \
                                   (max_value_sag[c_num]+ max_value_cor[c_num])
                # if max_value_sag[c_num] > max_value_cor[c_num]:
                #     max_idx_y[c_num] = max_idx_sag_y[c_num]
                # else:
                #     max_idx_y[c_num] = max_idx_cor_x[c_num]

        for step in range(len(threshold_label)):
            position_batch[step, i, :, 0] = (max_idx_x.float() * resolution / spacing[0][i] + crop_info['displace'][i, 0, 0])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 1] = (max_idx_y.float() * resolution / spacing[1][i] + crop_info['displace'][i, 0, 1])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 2] = (max_idx_z.float() * resolution / spacing[2][i] + crop_info['displace'][i, 0, 2])\
                                      * (2 * (max_value > threshold_label[step]).float() - 1) \
                                      - (max_value <= threshold_label[step]).float()
            position_batch[step, i, :, 3] = max_value

        position_batch_sag[i, :, 0] = max_idx_sag_x
        position_batch_sag[i, :, 1] = max_idx_sag_y
        position_batch_sag[i, :, 2] = max_value_sag
        position_batch_cor[i, :, 0] = max_idx_cor_x
        position_batch_cor[i, :, 1] = max_idx_cor_y
        position_batch_cor[i, :, 2] = max_value_cor

    return position_batch, position_batch_cor , position_batch_sag


@torch.no_grad()
def pred_pos_4(device, output_sag_batch, output_cor_batch, direction, crop_info, spacing, cor_pad, sag_pad):
    """
    Compute the tensor product between output_sag and output_cor,
    then use argmax to find the position of the ith vertebra in channel i.
    Let's say the original 3D data has a shape of (B, C, d0, d1, d2), normally with C=25.
    Parameters:
        output_sag & output_cor: output of the Btrfly Net
        direc: it should be ('Z', 'Y', 'X') or ('Y', 'X', 'Z'), indicating the direction of the subject
    Return:
        a (BxCx3) tensor about the positions of the bones of every subjects in the batch
    """
    if output_sag_batch.shape[:2] != output_cor_batch.shape[:2]:
        raise Exception("output_sag and output_cor have different batch sizes or channel numbers!")
    B, C = output_sag_batch.shape[0], output_sag_batch.shape[1]
    f_size = 7
    # threshold to reduce noise
    threshold_noise = 0
    threshold_label = torch.from_numpy(np.arange(0, 0.4, 0.01)).float()
    position_batch = torch.Tensor(len(threshold_label), B, C, 4)
    position_batch_sag = torch.Tensor(B, C, 2)
    position_batch_cor = torch.Tensor(B, C, 2)
    resolution = 1.0
    ori_d0, ori_d1, ori_d2 = np.zeros(B), np.zeros(B), np.zeros(B)
    for i in range(B):
        direc = (direction[0][i], direction[1][i], direction[2][i])
        if (direc != ('Z', 'Y', 'X')) & (direc != ('Y', 'X', 'Z')):
            raise Exception('Unknown direction!')
        # select ith subject
        output_cor = output_cor_batch[i, :, :, :]
        output_sag = output_sag_batch[i, :, :, :]

        # reduce the noise according to threshold
        reduce_noise_sag = torch.where(output_sag < threshold_noise, torch.full_like(output_sag, 0), output_sag)
        reduce_noise_cor = torch.where(output_cor < threshold_noise, torch.full_like(output_cor, 0), output_cor)
        max_value, max_idx, max_cor_num = torch.zeros(24), torch.zeros(24), torch.zeros(24)

        if direc == ('Z', 'Y', 'X'):
            # sag:(C, d1, d2), cor:(C, d0, d2)
            if (output_sag.shape[2] != output_cor.shape[2]):
                raise Exception("sag and cor should have an identical size in the last dimension!")
            d0, d1, d2 = output_cor.shape[1], output_sag.shape[1], output_sag.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - cor_pad[2][i] - cor_pad[3][i], d1 - sag_pad[2][i] - sag_pad[3][
                i], d2 - cor_pad[0][i] - cor_pad[1][i]

            reduce_noise_sag_no_padding = reduce_noise_sag[:, sag_pad[2][i]:d1 - sag_pad[3][i],
                                          sag_pad[0][i]:d2 - sag_pad[1][i]]
            reduce_noise_cor_no_padding = reduce_noise_cor[:, cor_pad[2][i]:d0 - cor_pad[3][i],
                                          cor_pad[0][i]:d2 - cor_pad[1][i]]

            # for c_num in range(24):
            #
            #     reduce_noise_sag_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_sag_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)
            #     reduce_noise_cor_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_cor_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)

            # (24)
            max_value_sag, max_idx_sag = torch.max(
                reduce_noise_sag_no_padding.contiguous().view(reduce_noise_sag_no_padding.shape[0], -1), dim=1)
            max_value_cor, max_idx_cor = torch.max(
                reduce_noise_cor_no_padding.contiguous().view(reduce_noise_cor_no_padding.shape[0], -1), dim=1)
            max_idx_sag_x, max_idx_sag_y = max_idx_sag // ori_d2[i], max_idx_sag % ori_d2[i]
            max_idx_cor_x, max_idx_cor_y = max_idx_cor // ori_d2[i], max_idx_cor % ori_d2[i]

            # extend them to (d0, d1, d2)
            for c_num_sag in range(24):
                reduce_noise_sag_one_cha = reduce_noise_sag_no_padding[c_num_sag, :, :].unsqueeze(0).expand(int(ori_d0[i]), int(ori_d1[i]), int(ori_d2[i]))
                for c_num_cor in range(24):
                    reduce_noise_cor_one_cha = reduce_noise_cor_no_padding[c_num_cor, :, :].unsqueeze(1).expand(int(ori_d0[i]), int(ori_d1[i]),
                                                                                            int(ori_d2[i]))

                    product = reduce_noise_cor_one_cha * reduce_noise_sag_one_cha

                    # find maximum value for each batch and channel
                    max_value_tmp, max_idx_tmp = torch.max(product.view(-1), dim=0)
                    if max_value_tmp.cpu() > max_value[c_num_sag]:
                        max_value[c_num_sag], max_idx[c_num_sag], max_cor_num[c_num_sag] = max_value_tmp, max_idx_tmp, c_num_cor
        else:
            # sag:(C, d0, d1), cor:(C, d1, d2)
            if (output_sag.shape[2] != output_cor.shape[1]):
                raise Exception("sag and cor should have an identical size in some dimension!")
            d0, d1, d2 = output_sag.shape[1], output_sag.shape[2], output_cor.shape[2]
            ori_d0[i], ori_d1[i], ori_d2[i] = d0 - sag_pad[2][i] - sag_pad[3][i], d1 - sag_pad[0][i] - sag_pad[1][
                i], d2 - cor_pad[0][i] - cor_pad[1][i]

            reduce_noise_sag_no_padding = reduce_noise_sag[:, sag_pad[2][i]:d1 - sag_pad[3][i],
                                          sag_pad[0][i]:d2 - sag_pad[1][i]]
            reduce_noise_cor_no_padding = reduce_noise_cor[:, cor_pad[2][i]:d0 - cor_pad[3][i],
                                          cor_pad[0][i]:d2 - cor_pad[1][i]]

            # for c_num in range(24):
            #
            #     reduce_noise_sag_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_sag_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)
            #     reduce_noise_cor_no_padding[c_num, :, :] = torch.tensor(cv2.medianBlur(reduce_noise_cor_no_padding[c_num, :, :].cpu().numpy(), f_size)).to(device)

            # (24)
            max_value_sag, max_idx_sag = torch.max(
                reduce_noise_sag_no_padding.contiguous().view(reduce_noise_sag_no_padding.shape[0], -1), dim=1)
            max_value_cor, max_idx_cor = torch.max(
                reduce_noise_cor_no_padding.contiguous().view(reduce_noise_cor_no_padding.shape[0], -1), dim=1)

            max_idx_sag_x, max_idx_sag_y = max_idx_sag // ori_d1[i], max_idx_sag % ori_d1[i]
            max_idx_cor_x, max_idx_cor_y = max_idx_cor // ori_d2[i], max_idx_cor % ori_d2[i]

            # extend them to (d0, d1, d2)
            for c_num_sag in range(24):
                reduce_noise_sag_one_cha = reduce_noise_sag_no_padding[c_num_sag, :, :].unsqueeze(2).expand(int(ori_d0[i]), int(ori_d1[i]),
                                                                                        int(ori_d2[i]))
                for c_num_cor in range(24):
                    reduce_noise_cor_one_cha = reduce_noise_cor_no_padding[c_num_cor, :, :].unsqueeze(0).expand(int(ori_d0[i]), int(ori_d1[i]),
                                                                                        int(ori_d2[i]))

                    product = reduce_noise_cor_one_cha * reduce_noise_sag_one_cha

                    # find maximum value for each batch and channel
                    max_value_tmp, max_idx_tmp = torch.max(product.view(-1), dim=0)
                    if max_value_tmp.cpu() > max_value[c_num_sag]:
                        max_value[c_num_sag], max_idx[c_num_sag], max_cor_num[c_num_sag] = max_value_tmp, max_idx_tmp, c_num_cor



        # translate the indexes to 3D form
        max_idx_x, max_idx_y, max_idx_z = -torch.ones(24), -torch.ones(24), -torch.ones(24)
        for c_num in range(24):
            max_idx_x[c_num], max_idx_y[c_num], max_idx_z[c_num] = \
                max_idx[c_num] // (ori_d1[i] * ori_d2[i]), \
                (max_idx[c_num] % (ori_d1[i] * ori_d2[i])) // ori_d2[i], \
                (max_idx[c_num] % (ori_d1[i] * ori_d2[i])) % ori_d2[i]

        if direc == ('Z', 'Y', 'X'):
            for c_num in range(24):
                max_idx_z[c_num] = (max_idx_sag_y[c_num] * max_value_sag[c_num] + max_idx_cor_y[c_num] * max_value_cor[
                    c_num]) / \
                                   (max_value_sag[c_num] + max_value_cor[c_num])
                # if max_value_sag[c_num] > max_value_cor[c_num]:
                #     max_idx_z[c_num] = max_idx_sag_y[c_num]
                # else:
                #     max_idx_z[c_num] = max_idx_cor_y[c_num]

        else:
            for c_num in range(24):
                max_idx_y[c_num] = (max_idx_sag_y[c_num] * max_value_sag[c_num] + max_idx_cor_x[c_num] * max_value_cor[
                    c_num]) / \
                                   (max_value_sag[c_num] + max_value_cor[c_num])
                # if max_value_sag[c_num] > max_value_cor[c_num]:
                #     max_idx_y[c_num] = max_idx_sag_y[c_num]
                # else:
                #     max_idx_y[c_num] = max_idx_cor_x[c_num]

        for step in range(len(threshold_label)):
            position_batch[step, i, :, 0] = (max_idx_x.float() * resolution / spacing[0][i] + crop_info['displace'][
                i, 0, 0]) \
                                            * (2 * (max_value >= threshold_label[step]).float() - 1) \
                                            - (max_value < threshold_label[step]).float()
            position_batch[step, i, :, 1] = (max_idx_y.float() * resolution / spacing[1][i] + crop_info['displace'][
                i, 0, 1]) \
                                            * (2 * (max_value >= threshold_label[step]).float() - 1) \
                                            - (max_value < threshold_label[step]).float()
            position_batch[step, i, :, 2] = (max_idx_z.float() * resolution / spacing[2][i] + crop_info['displace'][
                i, 0, 2]) \
                                            * (2 * (max_value >= threshold_label[step]).float() - 1) \
                                            - (max_value < threshold_label[step]).float()
            position_batch[step, i, :, 3] = max_value
        position_batch_sag[i, :, 0] = max_idx_sag_x
        position_batch_sag[i, :, 1] = max_idx_sag_y
        position_batch_cor[i, :, 0] = max_idx_cor_x
        position_batch_cor[i, :, 1] = max_idx_cor_y

    return position_batch, position_batch_cor, position_batch_sag