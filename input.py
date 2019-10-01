import numpy as np
import os
import nibabel as nib
import SimpleITK as sitk
import json
import scipy.io as io
import imageio


def get_verse_list(path):
    """
    get the list of the vertebrate data
    form eg---> verse006.nii
    :param path:
    :return:
    """
    file_list = []
    file_list_train = []
    file_list_vali = []
    file_list_train_pos = []
    file_list_vali_pos = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith('.'):
                continue
            else:
                if (file == 'verse113.nii') | (file == 'verse104.nii') | (file == 'verse201.nii'):
                    continue
                else:
                    file_list.append(os.path.join(file))


    wholesize = len(file_list)
    index = range(wholesize)
    train_len = wholesize
    vali_len = wholesize - train_len
    train_index = index[0:train_len]

    vali_index = index[train_len:]
    train_index = sorted(train_index)
    vali_index = sorted(vali_index)
    for idx in train_index:
        file_list_train.append(file_list[idx])
        file_list_train_pos.append(file_list[idx][0:-4] + '_ctd.json')
    for idx in vali_index:
        file_list_vali.append(file_list[idx])
        file_list_vali_pos.append(file_list[idx][0:-4] + '_ctd.json')
    file_list_train.sort()
    file_list_train_pos.sort()
    return file_list_train, file_list_train_pos, train_len

def get_centroid_pos(raw, w, h, c, fileJson ):
    """
    :param raw: from sitk
    :param w: from nib
    :param h:
    :param c:
    :param labelidx:
    :return:
    """
    Dic = {0: 'Z', 1: 'Y', 2: 'X'}
    direction = np.round(list(raw.GetDirection()))
    direc0 = direction[0:7:3]
    direc1 = direction[1:8:3]
    direc2 = direction[2:9:3]
    dim0char = Dic[(np.argwhere((np.abs(direc0 )) == 1))[0][0]]
    dim1char = Dic[(np.argwhere((np.abs(direc1 )) == 1))[0][0]]
    dim2char = Dic[(np.argwhere((np.abs(direc2 )) == 1))[0][0]]
    resolution = raw.GetSpacing()
    label = fileJson['label']
    if np.sum(direc0) == -1:
        if dim0char == 'X':
            dim0 = fileJson['X']/resolution[0]
        else:
            dim0 = w - fileJson[dim0char]/resolution[0]
    else:
        if dim0char == 'X':
            dim0 = w - fileJson['X']/resolution[0]
        else:
            dim0 = fileJson[dim0char]/resolution[0]

    if np.sum(direc1) == -1:
        if dim1char == 'X':
            dim1 = fileJson['X']/resolution[1]
        else:
            dim1 = h - fileJson[dim1char]/resolution[1]
    else:
        if dim1char == 'X':
            dim1 = h - fileJson['X']/resolution[1]
        else:
            dim1 = fileJson[dim1char]/resolution[1]

    if np.sum(direc2) == -1:
        if dim2char == 'X':
            dim2 = fileJson['X']/resolution[2]
        else:
            dim2 = c - fileJson[dim2char]/resolution[2]
    else:
        if dim2char == 'X':
            dim2 = c - fileJson['X']/resolution[2]
        else:
            dim2 = fileJson[dim2char]/resolution[2]

    return label, int(dim0), int(dim1), int(dim2)

def image_mode(img_path):
    Dic = {0:'Z', 1:'Y', 2:'X'}
    img = sitk.ReadImage(img_path)
    direction = np.round(list(img.GetDirection()))
    direc0 = direction[0:7:3]
    direc1 = direction[1:8:3]
    direc2 = direction[2:9:3]

    dim0_char = Dic[(np.argwhere((np.abs(np.round(direc0))) == 1))[0][0]]
    dim1_char = Dic[(np.argwhere((np.abs(np.round(direc1))) == 1))[0][0]]
    dim2_char = Dic[(np.argwhere((np.abs(np.round(direc2))) == 1))[0][0]]

    return dim0_char, dim1_char, dim2_char

def prepare_SSD_input(img_path='/shenlab/local/zhenghan/original/', is_test=False):
    """
    generate heat map and save it to "datasets/VOC2007/Label2D/" as .mat file
    :input:
        img_path: path of original directory
    """
    resolution = 2.0
    nii_path = 'test/' if is_test else 'raw/'
    raw_file_list, pos_file_list, file_num = get_verse_list(img_path + nii_path)
    for i in range(file_num):
        raw_file_name, pos_file_name = raw_file_list[i], pos_file_list[i]
        print(raw_file_name)

        direc = image_mode(img_path + nii_path + raw_file_name)
        if (direc != ('Z', 'Y', 'X')) & (direc != ('Y', 'X', 'Z')):
            raise Exception('Unknown direction!')

        img_handle_nib = nib.load(img_path + nii_path + raw_file_name)

        img_handle_sitk = sitk.ReadImage(img_path + 'raw/' + raw_file_name)
        spacing = img_handle_sitk.GetSpacing()
        w, h, c = img_handle_sitk.GetSize()

        #new_size_w = round(w * spacing[0] / resolution)
        #new_size_h = round(h * spacing[1] / resolution)
        #new_size_c = round(c * spacing[2] / resolution)
        new_size_w, new_size_h, new_size_c = w, h, c


        img_raw = img_handle_nib.get_fdata()
        #img_same_res = transform.resize(img_raw, (new_size_w, new_size_h, new_size_c))
        img_same_res = img_raw
        imageio.imwrite("../SSD/datasets/VOC2007/JPEGImages/" + raw_file_name[:-4] + "_0.jpg", np.max(img_same_res, axis=0))
        imageio.imwrite("../SSD/datasets/VOC2007/JPEGImages/" + raw_file_name[:-4] + "_1.jpg", np.max(img_same_res, axis=1))
        imageio.imwrite("../SSD/datasets/VOC2007/JPEGImages/" + raw_file_name[:-4] + "_2.jpg", np.max(img_same_res, axis=2))

def prepare_heat_map(img_path='/shenlab/local/zhenghan/original/'):
    resolution = 2.0
    raw_file_list, pos_file_list, file_num = get_verse_list(img_path + 'raw')
    for i in range(file_num):
        raw_file_name, pos_file_name = raw_file_list[i], pos_file_list[i]
        print(raw_file_name)

        direc = image_mode(img_path + 'raw/' + raw_file_name)
        if (direc != ('Z', 'Y', 'X')) & (direc != ('Y', 'X', 'Z')):
            raise Exception('Unknown direction!')

        img_handle_nib = nib.load(img_path + 'raw/' + raw_file_name)

        img_handle_sitk = sitk.ReadImage(img_path + 'raw/' + raw_file_name)
        spacing = img_handle_sitk.GetSpacing()
        w, h, c = img_handle_sitk.GetSize()

        if direc == ('Z', 'Y', 'X'):
            new_size_side_0 = new_size_h
            new_size_side_1 = new_size_c
            new_size_front_0 = new_size_w
            new_size_front_1 = new_size_c
        elif direc == ('Y', 'X', 'Z'):
            new_size_side_0 = new_size_w
            new_size_side_1 = new_size_h
            new_size_front_0 = new_size_h
            new_size_front_1 = new_size_c
        else:
            raise Exception('Unknown direction!')

        pos_file_handle = open(img_path + 'pos/' + pos_file_name, "rb")
        pos_file_json = json.load(pos_file_handle)

        label_side = np.zeros((new_size_side_0, new_size_side_1, 25))
        label_front = np.zeros((new_size_front_0, new_size_front_1, 25))

        x_side, y_side = np.meshgrid(range(new_size_side_0), range(new_size_side_1), indexing='ij')
        x_front, y_front = np.meshgrid(range(new_size_front_0), range(new_size_front_1), indexing='ij')

        for idx in pos_file_json:
            print(idx)
            label, location_x, location_y, location_z = get_centroid_pos(img_handle_sitk, w, h, c, idx)
            #location_x = location_x * spacing[0] / resolution
            #location_y = location_y * spacing[1] / resolution
            #location_z = location_z * spacing[2] / resolution
            if label == 25:
                continue
            if direc == ('Z', 'Y', 'X'):
                label_side[:, :, label] = np.exp(-((x_side - location_y) ** 2 + (y_side-location_z) ** 2) * 0.02)
                label_front[:, :, label] = np.exp(-((x_front - location_x) ** 2 + (y_front - location_z) ** 2) * 0.02)
            else:
                label_side[:, :, label] = np.exp(-((x_side - location_x) ** 2 + (y_side - location_y) ** 2) * 0.02)
                label_front[:, :, label] = np.exp(-((x_front - location_y) ** 2 + (y_front - location_z) ** 2) * 0.02)

        label_side[:, :, 0] = 1 - np.max(label_side[:, :, 1:25], axis=2)
        label_front[:, :, 0] = 1 - np.max(label_front[:, :, 1:25], axis=2)

        assert label_side.shape[0] in (w, h, c)
        assert label_side.shape[1] in (w, h, c)
        assert label_front.shape[0] in (w, h, c)
        assert label_front.shape[1] in (w, h, c)
        assert img_same_res.shape == (w, h, c)

        imageio.imwrite('datasets/Snapshot/' + raw_file_name[0:8] + '_side.jpg', np.sum(label_side[:, :, 1:25], axis=2))
        imageio.imwrite('datasets/Snapshot/' + raw_file_name[0:8] + '_front.jpg', np.sum(label_front[:, :, 1:25], axis=2))
        imageio.imwrite('datasets/Snapshot/' + raw_file_name[0:8] + '_side_bkgd.jpg', label_side[:, :, 0])
        imageio.imwrite('datasets/Snapshot/' + raw_file_name[0:8] + '_front_bkgd.jpg', label_front[:, :, 0])

        io.savemat('datasets/Label2D/' + raw_file_name[0:8] + ".mat",
                   {"side": label_side, "front": label_front})




if __name__ == '__main__':
    prepare_SSD_input()