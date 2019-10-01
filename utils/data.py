from scipy.io import loadmat
from input import *
import SimpleITK as sitk
import torch.nn.functional as func

def get_image_mode_length(img_path):
    Dic = {0:'Z', 1:'Y', 2:'X'}
    img = sitk.ReadImage(img_path)
    direction = np.round(list(img.GetDirection()))
    direc0 = direction[0:7:3]
    direc1 = direction[1:8:3]
    direc2 = direction[2:9:3]

    dim0_char = Dic[(np.argwhere((np.abs(np.round(direc0))) == 1))[0][0]]
    dim1_char = Dic[(np.argwhere((np.abs(np.round(direc1))) == 1))[0][0]]
    dim2_char = Dic[(np.argwhere((np.abs(np.round(direc2))) == 1))[0][0]]

    if [dim0_char, dim1_char, dim2_char] == ['Y', 'X', 'Z']:
        dimlength = 1
    elif [dim0_char, dim1_char, dim2_char] == ['Z', 'Y', 'X']:
        dimlength = 2

    return dimlength

class ProjectionDataset(Dataset.Dataset):
    def __init__(self, cfg, mat_dir=None, input_img_dir=None, transform=None):
        """
        :param mat_dir: (string) directory with all the .mat file containing heat maps
        :param input_img_dir: (string) directory with all the images
        :param transform: (callable, optional) optional transform to be applied on a sample
        """
        self.cfg = cfg
        self.mat_dir = mat_dir
        self.mat_list = glob.glob(pathname=os.path.join(mat_dir, "*.mat"))
        self.mat_list.sort()
        self.not_test = 'test' not in self.mat_list[0]

        self.input_img_dir = input_img_dir
        self.input_img_list = glob.glob(pathname=os.path.join(input_img_dir, "*.jpg"))
        self.input_img_list.sort()

        self.raw_file_dir = cfg.ORIGINAL_PATH + 'raw/' if self.not_test else cfg.ORIGINAL_PATH + 'test/'
        self.raw_file_list, self.pos_file_list, self.file_num = get_verse_list(self.raw_file_dir)

        self.crop_info_file_dir = cfg.CROP_INFO_DIR

        if "train" in input_img_dir:
            self.no_bg_file_dir = glob.glob(pathname=input_img_dir + "../train/*.jpg")
            self.no_bg_file_dir.sort()
        elif "val" in input_img_dir:
            self.no_bg_file_dir = glob.glob(pathname=input_img_dir + "../val/*.jpg")
            self.no_bg_file_dir.sort()
        else:
            pass

        if ("test" not in mat_dir) & (len(self.input_img_list) != 2 * len(self.mat_list)):
            raise Exception("Length error! The number of imgs should be 2 times the number of mat files.")

        self.transform = transform

    def __len__(self):
        return len(self.mat_list)

    def __getitem__(self, idx):
        res = 1.0
        not_test = 'test' not in self.mat_list[0]
        name = self.mat_list[idx][-12:-4]
        input_cor = imageio.imread(self.input_img_list[2 * idx])
        input_sag = imageio.imread(self.input_img_list[2 * idx + 1])

        label_2D_front = loadmat(self.mat_list[idx])["front"]
        label_2D_side = loadmat(self.mat_list[idx])["side"]
        raw_sitk = sitk.ReadImage(self.raw_file_dir + self.mat_list[idx][-12:-4] + ".nii")
        direction_sitk = raw_sitk.GetDirection()
        direction = image_mode(self.raw_file_dir + self.mat_list[idx][-12:-4] + ".nii")
        spacing = raw_sitk.GetSpacing()
        size_raw = raw_sitk.GetSize()
        crop_info = loadmat(self.crop_info_file_dir + self.mat_list[idx][-12:])

        sample = {'not_test': not_test, 'input_cor': input_cor, 'input_sag': input_sag, 'gt_cor': label_2D_front,
                'gt_sag': label_2D_side,
                'direction_sitk': direction_sitk, 'direction': direction, 'spacing': spacing,
                'size_raw': size_raw, 'crop_info': crop_info, 'name': name}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    def __call__(self, sample):
        input_cor, input_sag, gt_cor, gt_sag = sample['input_cor'], sample['input_sag'], sample['gt_cor'], sample['gt_sag']
        gt_cor = gt_cor.transpose((2, 0, 1)) if sample['not_test'] else gt_cor
        gt_sag = gt_sag.transpose((2, 0, 1)) if sample['not_test'] else gt_sag

        d0_uni = max(608, input_cor.shape[0], input_sag.shape[0])
        d1_uni = max(608, input_cor.shape[1], input_sag.shape[1])
        cor_pad_d0 = d0_uni - input_cor.shape[0]
        cor_pad_d1 = d1_uni - input_cor.shape[1]
        sag_pad_d0 = d0_uni - input_sag.shape[0]
        sag_pad_d1 = d1_uni - input_sag.shape[1]

        cor_pad = (cor_pad_d1 // 2 + (cor_pad_d1 % 2), cor_pad_d1 // 2, cor_pad_d0 // 2 + (cor_pad_d0 % 2), cor_pad_d0 // 2)
        sag_pad = (sag_pad_d1 // 2 + (sag_pad_d1 % 2), sag_pad_d1 // 2, sag_pad_d0 // 2 + (sag_pad_d0 % 2), sag_pad_d0 // 2)


        input_cor_padded = func.pad(torch.from_numpy(input_cor), cor_pad)
        input_sag_padded = func.pad(torch.from_numpy(input_sag), sag_pad)

        input_cor_padded = input_cor_padded.reshape((1, input_cor_padded.shape[0], input_cor_padded.shape[1]))
        input_sag_padded = input_sag_padded.reshape((1, input_sag_padded.shape[0], input_sag_padded.shape[1]))

        if sample['not_test']:
            gt_cor_padded = func.pad(torch.from_numpy(gt_cor), cor_pad)
            gt_sag_padded = func.pad(torch.from_numpy(gt_sag), sag_pad)

        assert input_cor_padded.shape[1] == d0_uni, (input_cor_padded.shape[1], d0_uni)
        assert input_cor_padded.shape[2] == d1_uni, (input_cor_padded.shape[2], d1_uni)
        assert input_sag_padded.shape[1] == d0_uni, (input_sag_padded.shape[1], d0_uni)
        assert input_sag_padded.shape[2] == d1_uni, (input_cor_padded.shape[2], d1_uni)

        if sample['not_test']:
            return {'input_cor':input_cor_padded, 'input_sag': input_sag_padded,
                    'gt_cor':gt_cor_padded, 'gt_sag':gt_sag_padded,
                    'direction_sitk':sample['direction_sitk'], 'direction':sample['direction'],
                    'spacing':sample['spacing'], 'size_raw':sample['size_raw'], 'crop_info':sample['crop_info'],
                    "cor_pad":cor_pad, "sag_pad":sag_pad, "name":sample["name"]}
        else:
            return {'input_cor':input_cor_padded, 'input_sag': input_sag_padded,
                    'direction_sitk':sample['direction_sitk'], 'direction':sample['direction'],
                    'spacing':sample['spacing'], 'size_raw':sample['size_raw'], 'crop_info':sample['crop_info'],
                    "cor_pad":cor_pad, "sag_pad":sag_pad, "name":sample["name"]}


