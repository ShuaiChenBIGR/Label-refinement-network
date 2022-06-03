import time
from collections import defaultdict
from tqdm import trange
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import src.py.training_on_the_fly.eval_val_3D as eval_val
import src.py.utils.one_hot_embedding as one_hot
from src.py.utils.vis_loss import history_log, visualize_loss
from src.py.utils.visualize_batch import visualize_main_seg
import src.py.utils.mkdir as mkdir
import torch.optim as optim
from src.py.utils.loss import DiceCoefficientLF
from src.py.networks.unet_3D import *
import random

from time import time
import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.transforms.abstract_transforms import Compose
from scipy import ndimage as ndi
from skimage.morphology import cube


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs)**exponent


class AirwayDataLoader3D(DataLoader):
    def __init__(self, data, batchsize, patch_size, num_threads_in_multithreaded, job_description, seed_for_shuffle=1234, reture_incomplete=False, shuffle=True, crop='random', modalities=1, mode='train'):
        super().__init__(data, batchsize, num_threads_in_multithreaded, seed_for_shuffle, reture_incomplete, shuffle, True)
        self.patch_size = patch_size
        self.num_modalities = modalities
        self.indices = list(range(len(data)))
        self.crop = crop
        self.job_description = job_description
        self.mode = mode

    @staticmethod
    def load_patient(patient):
        data = np.load(patient + '.npy', mmap_mode='r')
        metadata = load_pickle(patient + '.pkl')
        return data, metadata

    def generate_train_batch(self):
        idx = self.get_indices()
        patient_for_batch = [self._data[i] for i in idx]

        data = np.zeros((self.batch_size, self.num_modalities, *self.patch_size), dtype=np.float32)
        if self.job_description['dataset'] == 'airway':
            seg = np.zeros((self.batch_size, 2, *self.patch_size), dtype=np.float32)
        else:
            seg = np.zeros((self.batch_size, 1, *self.patch_size), dtype=np.float32)

        metadata = []
        patient_names = []

        for i, j in enumerate(patient_for_batch):
            patient_data, patient_metadata = self.load_patient(j)

            if self.job_description['dataset'] == 'airway':
                patient_data = pad_nd_image(patient_data, self.patch_size)
                patient_data, patient_seg = crop(patient_data[:-2][None], patient_data[-2:][None], self.patch_size,
                                                 crop_type=self.crop)
            else:

                # now find the nonzero region and crop to that
                nonzero = [np.array(np.where(i != 0)) for i in patient_data[:-1]]
                nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
                nonzero = np.array([np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]).T
                # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis

                # now crop to nonzero
                patient_data = patient_data[:,
                           nonzero[0, 0] : nonzero[0, 1] + 1,
                           nonzero[1, 0]: nonzero[1, 1] + 1,
                           nonzero[2, 0]: nonzero[2, 1] + 1,
                           ]

                patient_data, patient_seg = crop(patient_data[:-1][None], patient_data[-1:][None], self.patch_size, crop_type=self.crop)

                patient_seg[0, 0] = ndi.binary_dilation(patient_seg[0, 0], cube(3))

            data[i] = patient_data[0]
            seg[i] = patient_seg[0]

            metadata.append(patient_metadata)
            patient_names.append(j)

        return {'data': data, 'seg': seg, 'metadata': metadata, 'names': patient_names}

def get_train_transform(patch_size):

    tr_transforms = []
    tr_transforms.append(
        SpatialTransform_2(
            patch_size, [i // 2 for i in patch_size],
            do_elastic_deform=False, deformation_scale=(0, 0.25),
            do_rotation=True,
            angle_x=(- 30 / 360. * 2 * np.pi, 30 / 360. * 2 * np.pi),
            angle_y=(- 30 / 360. * 2 * np.pi, 30 / 360. * 2 * np.pi),
            angle_z=(- 30 / 360. * 2 * np.pi, 30 / 360. * 2 * np.pi),
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data='constant', border_cval_data=0,
            border_mode_seg='constant', border_cval_seg=0,
            order_seg=1, order_data=3,
            random_crop=True,
            p_el_per_sample=0.0, p_rot_per_sample=0.2, p_scale_per_sample=0.2
        )
    )

    tr_transforms.append(MirrorTransform(axes=(0, 1, 2)))
    tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, p_per_sample=0.3))
    tr_transforms = Compose(tr_transforms)
    return tr_transforms


def start_main_3D(job_description):

    # initialization
    model = UNet_3D(job_description=job_description, skip=True, LR=False).to(job_description['device'])

    best_val = 0
    best_epoch = 0
    dict = defaultdict(list)

    mkdir.mkdir(job_description['result_path'] + '/model')
    mkdir.mkdir(job_description['result_path'] + '/preview_main/train')
    mkdir.mkdir(job_description['result_path'] + '/preview_main/val')

    # Dataloader setting
    patch_size = job_description['Patch_size']
    batch_size = job_description['batch_size']

    shapes = [AirwayDataLoader3D.load_patient(i)[0].shape[1:] for i in job_description['patients_keys']]
    max_shape = np.max(shapes, 0)
    max_shape = np.max((max_shape, patch_size), 0)

    dataloader_train = AirwayDataLoader3D(job_description['train_keys'], batch_size, max_shape, 1, job_description, seed_for_shuffle=job_description['seed'], modalities=1, mode='train')
    dataloader_validation = AirwayDataLoader3D(job_description['val_keys'], batch_size, patch_size, 1, job_description, seed_for_shuffle=job_description['seed'], shuffle=False, crop='center', modalities=1, mode='valid')
    tr_transforms = get_train_transform(patch_size)

    tr_gen = MultiThreadedAugmenter(dataloader_train, tr_transforms, num_processes=4,
                                    num_cached_per_queue=1,
                                    seeds=None, pin_memory=False)

    val_gen = MultiThreadedAugmenter(dataloader_validation, None,
                                     num_processes=4, num_cached_per_queue=1,
                                     seeds=None,
                                     pin_memory=False)
    tr_gen.restart()
    val_gen.restart()

    num_batches_per_epoch = job_description['number_training_batches_per_epoch']
    num_validation_batches_per_epoch = job_description['number_validation_batches_per_epoch']

    time_per_epoch = []
    start = time()

    criterion = (DiceCoefficientLF(),)
    eval_mode = eval_val.eval_val_seg_3D(job_description=job_description)
    tqiter = trange(job_description['epoch']-job_description['epoch_restart']-1, desc=job_description['dataset'])

    epoch_val = 0
    best_val = epoch_val

    for epoch in tqiter:
        epoch = epoch + job_description['epoch_restart']
        start_epoch = time()
        fig_loss = plt.figure(num='loss', figsize=[10, 3.8])
        epoch_train = 0

        if job_description['pretrain_main']:
            model.load_state_dict(torch.load(job_description['result_path'] + '/model/best_val.pth'))

        for b in range(num_batches_per_epoch):
            model.train()

            if job_description['dataset'] == 'airway':
                balance = random.uniform(0, 1)

            if balance > 0.3:
                while True:
                    batch = next(tr_gen)
                    if job_description['dataset'] == 'airway':
                        inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                        labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 0:1]
                        lungs = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 1:2]
                    else:
                        inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                        labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)
                    print('lungs:', torch.unique(lungs))
                    print('gt:', torch.unique(labels))
                    if len(torch.unique(lungs)) > 1:
                        break
            else:
                batch = next(tr_gen)
                if job_description['dataset'] == 'airway':
                    inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                    labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 0:1]
                    lungs = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 1:2]
                else:
                    inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                    labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)

            labels = one_hot.one_hot_embedding(labels, num_classes=2)
            labels = labels[:, 0].permute(0, 4, 1, 2, 3).to(job_description['device'], dtype=torch.float)

            optimizer = (optim.SGD(model.parameters(), lr=poly_lr(epoch, job_description['epoch'], job_description['lr'], 0.9), momentum=0.99, nesterov=True, weight_decay=3e-05), )
            optimizer[0].zero_grad()

            outputs_seg = model.forward(inputs)

            if job_description['dataset'] == 'airway':
                outputs_seg = outputs_seg * lungs

            loss = criterion[0](outputs_seg[:, :], labels[:, 1:])
            loss.backward()
            optimizer[0].step()
            epoch_train += loss.item()

            # visualize training set at the end of each epoch
            if epoch % job_description['vis_train'] == job_description['vis_train']-1:
                if job_description['preview_train'] == True:
                    if b == 0:
                        slice = int(job_description['Patch_size'][0]/2)
                        inputs_vis = inputs.cpu().detach().numpy()[0][0][slice]
                        label_vis = labels.cpu().detach().numpy()[0][1][slice]
                        outputs_vis = outputs_seg.cpu().detach().numpy()[0][0][slice]

                        fig_batch = plt.figure(figsize=[7, 10])
                        visualize_main_seg(inputs_vis, label_vis, outputs_vis, epoch=epoch)

                        plt.savefig(job_description['result_path'] + '/preview_main/train/' + 'epoch_%s.jpg' % epoch)
                        plt.close(fig_batch)

        tqiter.set_description(job_description['dataset'] + '(train=%.4f, val=%.4f)'
                               % (epoch_train, epoch_val))

        if epoch % job_description['val_epoch'] == job_description['val_epoch'] - 1:
            epoch_val = eval_mode.eval_val(model, val_gen, num_validation_batches_per_epoch, epoch, job_description)

            if epoch_val > best_val:
                best_val = epoch_val
                best_epoch = epoch
                torch.save(model.state_dict(), job_description['result_path'] + '/model/best_val.pth')

        # save and visualize training information
        if epoch == 0:
            title = 'Epoch     Train     Val'    'best_epoch\n'
            history_log(job_description['result_path'] + '/history_log.txt', title, 'w')
            history = (
                '{:3d}        {:.4f}       {:.4f}       {:d}\n'
                    .format(epoch, epoch_train / (num_batches_per_epoch),  epoch_val, best_epoch))
            history_log(job_description['result_path'] + '/history_log.txt', history, 'a')

            title = title.split()
            data = history.split()
            for ii, key in enumerate(title[:]):
                if ii == 0:
                    dict[key].append(int(data[ii]))
                else:
                    dict[key].append(float(data[ii]))
            visualize_loss(fig_loss, dict=dict, title=title, epoch=epoch)
            plt.savefig(job_description['result_path'] + '/Log.jpg')
            plt.close(fig_loss)

        elif epoch > 0:
            title = 'Epoch     Train     Val'    'best_epoch\n'
            history = (
                '{:3d}        {:.4f}       {:.4f}       {:d}\n'
                    .format(epoch, epoch_train / (num_batches_per_epoch),  epoch_val, best_epoch))
            history_log(job_description['result_path'] + '/history_log.txt', history, 'a')

            title = title.split()
            data = history.split()
            for ii, key in enumerate(title[:]):
                if ii == 0:
                    dict[key].append(int(data[ii]))
                else:
                    dict[key].append(float(data[ii]))
            visualize_loss(fig_loss, dict=dict, title=title, epoch=epoch)
            plt.savefig(job_description['result_path'] + '/Log.jpg')
            plt.close(fig_loss)

        end_epoch = time()
        time_per_epoch.append(end_epoch - start_epoch)

    end = time()
    total_time = end - start
    print('-' * 64)
    print("Running %d epochs took a total of %.2f seconds with time per epoch being %s" %
          (job_description['epoch'], total_time, str(time_per_epoch)))
    print('main training finished')
    print('-' * 64)
