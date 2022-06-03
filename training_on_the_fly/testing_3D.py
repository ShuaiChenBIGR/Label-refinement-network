from src.py.utils.loss import *
import src.py.utils.mkdir as mkdir
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from src.py.networks.unet_3D import *
from src.py.networks.Adv import *
from src.py.networks.DoubleUNet import *
from src.py.networks.LR_3D import *
from src.py.networks.GAN_3D import *
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import *


def unpad_nd_image_3D(data, new_shape, old_shape):

    if new_shape[0] > old_shape[0]:
        difference = abs(new_shape[0] - old_shape[0])
        lower = difference // 2
        upper = new_shape[0] - (difference // 2) - difference % 2
        data = data[:, lower:upper]

    if new_shape[1] > old_shape[1]:
        difference = abs(new_shape[1] - old_shape[1])
        lower = difference // 2
        upper = new_shape[1] - (difference // 2) - difference % 2
        data = data[:, :, lower:upper]

    if new_shape[2] > old_shape[2]:
        difference = abs(new_shape[2] - old_shape[2])
        lower = difference // 2
        upper = new_shape[2] - (difference // 2) - difference % 2
        data = data[:, :, :, lower:upper]

    return data


def testing_seg_3D(job_description):

    model = UNet_3D(job_description, skip=True, LR=False).to(job_description['device'])
    acc_overall = np.zeros([len(job_description['test_keys']), 1])
    acc_avg = 0
    dice_all = 0

    mkdir.mkdir(job_description['result_path'] + '/test_cnn/nii')
    mkdir.mkdir(job_description['result_path'] + '/test_cnn/nii_labels')
    mkdir.mkdir(job_description['result_path'] + '/test_cnn/npy')

    file_acc = open(job_description['result_path'] + '/test_cnn' + '/accuracy_summary.txt', 'w')  # Avg Dice for all images for a particular class
    file_all = open(job_description['result_path'] + '/test_cnn' + '/seg_all_list.txt', 'w')

    model.eval()
    model.load_state_dict(torch.load(job_description['result_path'] + '/model/best_val.pth'))

    keys = job_description['test_keys']
    if job_description['train_valid_initial_test']:
        if job_description['test_train_keys'] == 'train':
            keys = job_description['train_keys']
        elif job_description['test_train_keys'] == 'val':
            keys = job_description['val_keys']
        elif job_description['test_train_keys'] == 'unlabel':
            keys = job_description['unlabel_keys']

    for i, patient in enumerate(tqdm(keys, desc='testing')):
        data_raw = np.expand_dims(np.load(patient + '.npy'), axis=0)

        data = pad_nd_image(data_raw, job_description['Pad_size'])

        if job_description['dataset'] == 'airway':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            labels = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.long)
            # lungs = torch.from_numpy(data[:, 2:3]).to(job_description['device'], dtype=torch.long)
        if job_description['dataset'] == 'vessel':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            labels = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.float)
            # labels = one_hot.one_hot_embedding(labels, num_classes=2)
            # labels = labels.permute(0, 4, 1, 2, 3).to(job_description['device'], dtype=torch.float)[:, 1:2]

        # create prediction tensor
        prediction = torch.zeros(labels.size())
        imgShape = inputs.size()[2:]
        resultShape = job_description['Patch_size']

        imgMatrix = np.zeros([imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
        resultMatrix = np.ones([resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)

        overlapZ = 0.5
        overlapH = 0.5
        overlapW = 0.5

        interZ = int(resultShape[0] * (1.0 - overlapZ))
        interH = int(resultShape[1] * (1.0 - overlapH))
        interW = int(resultShape[2] * (1.0 - overlapW))

        iterZ = int(((imgShape[0] - resultShape[0]) / interZ) + 1)
        iterH = int(((imgShape[1] - resultShape[1]) / interH) + 1)
        iterW = int(((imgShape[2] - resultShape[2]) / interW) + 1)

        freeZ = imgShape[0] - (resultShape[0] + interZ * (iterZ - 1))
        freeH = imgShape[1] - (resultShape[1] + interH * (iterH - 1))
        freeW = imgShape[2] - (resultShape[2] + interW * (iterW - 1))

        startZ = int(freeZ / 2)
        startH = int(freeH / 2)
        startW = int(freeW / 2)

        total_patches = iterZ*iterH*iterW

        for z in range(0, iterZ):
            for h in range(0, iterH):
                for w in range(0, iterW):
                    input = inputs[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                            (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                            (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))].to(job_description['device'])

                    with torch.no_grad():
                        outputs_seg = model(input)

                    prediction[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += outputs_seg.to('cpu')
                    imgMatrix[ (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += resultMatrix

        imgMatrix = np.where(imgMatrix == 0, 1, imgMatrix)
        labels_np = labels.cpu().detach().numpy()[0, :]
        outputs_np = np.divide(prediction.cpu().detach().numpy()[0, :], imgMatrix)
        outputs_np = np.where(outputs_np > 0.5, 1, 0)

        # if job_description['dataset'] == 'airway':
            # outputs_np = outputs_np * lungs.cpu().detach().numpy()[0, 0]

        # dice for single class
        if job_description['dataset'] == 'vessel':

            for c in range(0, 1):
                if i == 0:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'w')
                else:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'a')

                dice = 0
                if np.max(labels_np[c]) == 1:
                    dice = Dice_numpy(outputs_np[c], labels_np[c])
                    file.write('{:.4f}\n'.format(dice))
                elif np.max(labels_np[c]) == 0:
                    dice = 0
                    file.write('None\n')
                file.close()
                acc_overall[i, c] = dice

            # dice for all classes
            dice = Dice_numpy(outputs_np, labels_np)
            file_all.write('{:.4f}\n'.format(dice))
            dice_all += dice

        # save files
        outputs_np_ready = unpad_nd_image_3D(outputs_np, job_description['Pad_size'], data_raw.shape[2:]).astype(float)
        labels_np_ready = unpad_nd_image_3D(labels_np, job_description['Pad_size'], data_raw.shape[2:]).astype(float)
        
        
        for c in range(outputs_np.shape[0]):

            metadata_np = load_pickle(patient + '.pkl')

            original_shape = metadata_np['original_shape']
            seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
            # nonzero = metadata_np['nonzero_region']
            # seg_original_shape[nonzero[0, 0]: nonzero[0, 1] + 1,
            # nonzero[1, 0]: nonzero[1, 1] + 1,
            # nonzero[2, 0]: nonzero[2, 1] + 1] = outputs_np_ready[c]
            seg_original_shape[:] = outputs_np_ready[c]
            np.save(join(job_description['result_path'] + '/test_cnn/npy/', patient.split(sep='/')[-1] + '_{:s}.npy').format(str(c)), seg_original_shape)
            sitk_output = sitk.GetImageFromArray(outputs_np_ready[c])
            sitk_output.SetDirection(metadata_np['direction'])
            sitk_output.SetOrigin(metadata_np['origin'])
            sitk_output.SetSpacing(tuple(metadata_np['spacing'][[2, 1, 0]]))
            sitk.WriteImage(sitk_output,
                            job_description['result_path'] + '/test_cnn/nii/' + patient.split(sep='/')[-1] + '_{:s}.nii.gz'.format(str(c)))
                            
            sitk_output = sitk.GetImageFromArray(labels_np_ready[c])
            sitk_output.SetDirection(metadata_np['direction'])
            sitk_output.SetOrigin(metadata_np['origin'])
            sitk_output.SetSpacing(tuple(metadata_np['spacing'][[2, 1, 0]]))
            sitk.WriteImage(sitk_output,
                            job_description['result_path'] + '/test_cnn/nii_labels/' + patient.split(sep='/')[-1] + '_{:s}_labels.nii.gz'.format(str(c)))

    for i in range(0, 1):
        acc = np.sum(acc_overall[:, i]) / len(keys)
        print('seg {:d}: {:.4f}'.format(i, acc))
        acc_avg += acc
        file_acc.write('{:.4f}\n'.format(acc))

    acc_avg = dice_all / len(keys)
    file_acc.write('\n{:.4f}\n'.format(acc_avg))
    file_acc.close()

    file_all.close()
    print('seg all: {:.4f}'.format(acc_avg))

    print('-' * 64)

    return None


def testing_adv_3D(job_description):

    model = Adv(job_description, testing=True).to(job_description['device'])
    acc_overall = np.zeros([len(job_description['test_keys']), 1])
    acc_avg = 0
    dice_all = 0

    mkdir.mkdir(job_description['result_path'] + '/test_cnn/nii')
    file_acc = open(job_description['result_path'] + '/test_cnn' + '/accuracy_summary.txt', 'w')  # Avg Dice for all images for a particular class
    file_all = open(job_description['result_path'] + '/test_cnn' + '/seg_all_list.txt', 'w')

    model.eval()
    model.load_state_dict(torch.load(job_description['result_path'] + '/model/best_val.pth'))

    for i, patient in enumerate(tqdm(job_description['test_keys'], desc='testing')):
        data_raw = np.expand_dims(np.load(patient + '.npy'), axis=0)

        data = pad_nd_image(data_raw, job_description['Pad_size'])

        if job_description['dataset'] == 'airway':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            labels = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.long)
            # lungs = torch.from_numpy(data[:, 2:3]).to(job_description['device'], dtype=torch.long)
        if job_description['dataset'] == 'vessel':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            labels = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.float)

        # create prediction tensor
        prediction = torch.zeros(labels.size())
        imgShape = inputs.size()[2:]
        resultShape = job_description['Patch_size']

        imgMatrix = np.zeros([imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
        resultMatrix = np.ones([resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)

        overlapZ = 0.5
        overlapH = 0.5
        overlapW = 0.5

        interZ = int(resultShape[0] * (1.0 - overlapZ))
        interH = int(resultShape[1] * (1.0 - overlapH))
        interW = int(resultShape[2] * (1.0 - overlapW))

        iterZ = int(((imgShape[0] - resultShape[0]) / interZ) + 1)
        iterH = int(((imgShape[1] - resultShape[1]) / interH) + 1)
        iterW = int(((imgShape[2] - resultShape[2]) / interW) + 1)

        freeZ = imgShape[0] - (resultShape[0] + interZ * (iterZ - 1))
        freeH = imgShape[1] - (resultShape[1] + interH * (iterH - 1))
        freeW = imgShape[2] - (resultShape[2] + interW * (iterW - 1))

        startZ = int(freeZ / 2)
        startH = int(freeH / 2)
        startW = int(freeW / 2)

        total_patches = iterZ*iterH*iterW

        for z in range(0, iterZ):
            for h in range(0, iterH):
                for w in range(0, iterW):
                    input = inputs[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                            (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                            (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))].to(job_description['device'])

                    label = labels[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                            (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                            (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))].to(job_description['device'])

                    with torch.no_grad():
                        cla, selection, outputs_seg = model(input, label[:, :])

                    prediction[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += outputs_seg.to('cpu')
                    imgMatrix[ (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += resultMatrix

        imgMatrix = np.where(imgMatrix == 0, 1, imgMatrix)
        if job_description['dataset'] == 'vessel':
            labels_np = labels.cpu().detach().numpy()[0, :]
        outputs_np = np.divide(prediction.cpu().detach().numpy()[0, :], imgMatrix)
        outputs_np = np.where(outputs_np > 0.5, 1, 0)

        outputs_np_ready = np.divide(prediction.cpu().detach().numpy()[0, :], imgMatrix)

        # dice for single class
        if job_description['dataset'] == 'vessel':
            for c in range(0, 1):
                if i == 0:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'w')
                else:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'a')

                dice = 0
                if np.max(labels_np[c]) == 1:
                    dice = Dice_numpy(outputs_np[c], labels_np[c])
                    file.write('{:.4f}\n'.format(dice))
                elif np.max(labels_np[c]) == 0:
                    dice = 0
                    file.write('None\n')
                file.close()
                acc_overall[i, c] = dice

            # dice for all classes
            dice = Dice_numpy(outputs_np, labels_np)
            file_all.write('{:.4f}\n'.format(dice))
            dice_all += dice

        # save files
        outputs_np_ready = unpad_nd_image_3D(outputs_np, job_description['Pad_size'], data_raw.shape[2:]).astype(float)
        for c in range(outputs_np.shape[0]):

            metadata_np = load_pickle(patient + '.pkl')

            original_shape = metadata_np['original_shape']
            seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
            # nonzero = metadata_np['nonzero_region']
            # seg_original_shape[nonzero[0, 0]: nonzero[0, 1] + 1,
            # nonzero[1, 0]: nonzero[1, 1] + 1,
            # nonzero[2, 0]: nonzero[2, 1] + 1] = outputs_np_ready[c]
            seg_original_shape[:] = outputs_np_ready[c]
            sitk_output = sitk.GetImageFromArray(seg_original_shape)
            sitk_output.SetDirection(metadata_np['direction'])
            sitk_output.SetOrigin(metadata_np['origin'])
            sitk_output.SetSpacing(tuple(metadata_np['spacing'][[2, 1, 0]]))
            sitk.WriteImage(sitk_output,
                            job_description['result_path'] + '/test_cnn/nii/' + patient.split(sep='/')[-1] + '_{:s}.nii.gz'.format(str(c)))

    for i in range(0, 1):
        acc = np.sum(acc_overall[:, i]) / len(job_description['test_keys'])
        print('seg {:d}: {:.4f}'.format(i, acc))
        acc_avg += acc
        file_acc.write('{:.4f}\n'.format(acc))

    acc_avg = dice_all / len(job_description['test_keys'])
    file_acc.write('\n{:.4f}\n'.format(acc_avg))
    file_acc.close()

    file_all.close()
    print('seg all: {:.4f}'.format(acc_avg))

    print('-' * 64)

    return None


def testing_dou_3D(job_description):

    model = DoubleUNet(job_description).to(job_description['device'])
    acc_overall = np.zeros([len(job_description['test_keys']), 1])
    acc_avg = 0
    dice_all = 0

    mkdir.mkdir(job_description['result_path'] + '/test_cnn/nii')
    file_acc = open(job_description['result_path'] + '/test_cnn' + '/accuracy_summary.txt', 'w')  # Avg Dice for all images for a particular class
    file_all = open(job_description['result_path'] + '/test_cnn' + '/seg_all_list.txt', 'w')

    model.eval()
    model.load_state_dict(torch.load(job_description['result_path'] + '/model/best_val.pth'))

    for i, patient in enumerate(tqdm(job_description['test_keys'], desc='testing')):
        data_raw = np.expand_dims(np.load(patient + '.npy'), axis=0)

        data = pad_nd_image(data_raw, job_description['Pad_size'])

        if job_description['dataset'] == 'airway':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            labels = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.long)
            # lungs = torch.from_numpy(data[:, 2:3]).to(job_description['device'], dtype=torch.long)
        if job_description['dataset'] == 'vessel':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            labels = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.float)

        # create prediction tensor
        prediction = torch.zeros(labels.size())
        imgShape = inputs.size()[2:]
        resultShape = job_description['Patch_size']

        imgMatrix = np.zeros([imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
        resultMatrix = np.ones([resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)

        overlapZ = 0.5
        overlapH = 0.5
        overlapW = 0.5

        interZ = int(resultShape[0] * (1.0 - overlapZ))
        interH = int(resultShape[1] * (1.0 - overlapH))
        interW = int(resultShape[2] * (1.0 - overlapW))

        iterZ = int(((imgShape[0] - resultShape[0]) / interZ) + 1)
        iterH = int(((imgShape[1] - resultShape[1]) / interH) + 1)
        iterW = int(((imgShape[2] - resultShape[2]) / interW) + 1)

        freeZ = imgShape[0] - (resultShape[0] + interZ * (iterZ - 1))
        freeH = imgShape[1] - (resultShape[1] + interH * (iterH - 1))
        freeW = imgShape[2] - (resultShape[2] + interW * (iterW - 1))

        startZ = int(freeZ / 2)
        startH = int(freeH / 2)
        startW = int(freeW / 2)

        total_patches = iterZ*iterH*iterW

        for z in range(0, iterZ):
            for h in range(0, iterH):
                for w in range(0, iterW):
                    input = inputs[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                            (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                            (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))].to(job_description['device'])

                    with torch.no_grad():
                        outputs_seg = model(input)

                    prediction[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += outputs_seg.to('cpu')
                    imgMatrix[ (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += resultMatrix

        imgMatrix = np.where(imgMatrix == 0, 1, imgMatrix)
        if job_description['dataset'] == 'vessel':
            labels_np = labels.cpu().detach().numpy()[0, :]
        outputs_np = np.divide(prediction.cpu().detach().numpy()[0, :], imgMatrix)
        outputs_np = np.where(outputs_np > 0.5, 1, 0)

        # dice for single class
        if job_description['dataset'] == 'vessel':
            for c in range(0, 1):
                if i == 0:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'w')
                else:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'a')

                dice = 0
                if np.max(labels_np[c]) == 1:
                    dice = Dice_numpy(outputs_np[c], labels_np[c])
                    file.write('{:.4f}\n'.format(dice))
                elif np.max(labels_np[c]) == 0:
                    dice = 0
                    file.write('None\n')
                file.close()
                acc_overall[i, c] = dice

            # dice for all classes
            dice = Dice_numpy(outputs_np, labels_np)
            file_all.write('{:.4f}\n'.format(dice))
            dice_all += dice

        # save files
        outputs_np_ready = unpad_nd_image_3D(outputs_np, job_description['Pad_size'], data_raw.shape[2:]).astype(float)
        for c in range(outputs_np.shape[0]):

            metadata_np = load_pickle(patient + '.pkl')

            original_shape = metadata_np['original_shape']
            seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
            # nonzero = metadata_np['nonzero_region']
            # seg_original_shape[nonzero[0, 0]: nonzero[0, 1] + 1,
            # nonzero[1, 0]: nonzero[1, 1] + 1,
            # nonzero[2, 0]: nonzero[2, 1] + 1] = outputs_np_ready[c]
            seg_original_shape[:] = outputs_np_ready[c]
            sitk_output = sitk.GetImageFromArray(seg_original_shape)
            sitk_output.SetDirection(metadata_np['direction'])
            sitk_output.SetOrigin(metadata_np['origin'])
            sitk_output.SetSpacing(tuple(metadata_np['spacing'][[2, 1, 0]]))
            sitk.WriteImage(sitk_output,
                            job_description['result_path'] + '/test_cnn/nii/' + patient.split(sep='/')[-1] + '_{:s}.nii.gz'.format(str(c)))

    for i in range(0, 1):
        acc = np.sum(acc_overall[:, i]) / len(job_description['test_keys'])
        print('seg {:d}: {:.4f}'.format(i, acc))
        acc_avg += acc
        file_acc.write('{:.4f}\n'.format(acc))

    acc_avg = dice_all / len(job_description['test_keys'])
    file_acc.write('\n{:.4f}\n'.format(acc_avg))
    file_acc.close()

    file_all.close()
    print('seg all: {:.4f}'.format(acc_avg))

    print('-' * 64)

    return None


def testing_lr_3D(job_description):

    model = LR_3D(job_description).to(job_description['device'])
    acc_overall = np.zeros([len(job_description['test_keys']), 1])
    acc_avg = 0
    dice_all = 0

    mkdir.mkdir(job_description['result_path'] + '/test_cnn/nii')
    mkdir.mkdir(job_description['result_path'] + '/test_cnn/npy')

    file_acc = open(job_description['result_path'] + '/test_cnn' + '/accuracy_summary.txt', 'w')  # Avg Dice for all images for a particular class
    file_all = open(job_description['result_path'] + '/test_cnn' + '/seg_all_list.txt', 'w')

    model.eval()
    model.load_state_dict(torch.load(job_description['result_path'] + '/model/best_val.pth'))

    keys = job_description['test_keys']
    initial_keys = job_description['test_initial_keys']

    if job_description['train_valid_initial_test']:
        if job_description['test_train_keys'] == 'train':
            keys = job_description['train_keys']
            initial_keys = job_description['train_initial_keys']
        elif job_description['test_train_keys'] == 'val':
            keys = job_description['val_keys']
            initial_keys = job_description['val_initial_keys']
        elif job_description['test_train_keys'] == 'unlabel':
            keys = job_description['unlabel_keys']
            initial_keys = job_description['unlabel_initial_keys']

    for i, patient in enumerate(tqdm(keys, desc='testing')):
        data_raw = np.expand_dims(np.load(patient + '.npy'), axis=0)
        data_initial = np.expand_dims(np.expand_dims(np.load(initial_keys[i] + '.npy'), axis=0), axis=0)
        data_raw = np.concatenate([data_raw, data_initial], axis=1)

        if job_description['dataset'] == 'airway':
            data_raw = np.concatenate([data_raw, data_initial], axis=1)

        data = pad_nd_image(data_raw, job_description['Pad_size'])

        if job_description['dataset'] == 'airway':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            gt = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.long)
            initials = torch.from_numpy(data[:, 2:3]).to(job_description['device'], dtype=torch.float)
        if job_description['dataset'] == 'vessel':
            inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
            gt = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.long)
            initials = torch.from_numpy(data[:, 2:3]).to(job_description['device'], dtype=torch.float)

        # create prediction tensor
        prediction = torch.zeros(gt.size())
        imgShape = inputs.size()[2:]
        resultShape = job_description['Patch_size']

        imgMatrix = np.zeros([imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
        resultMatrix = np.ones([resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)

        overlapZ = 0.5
        overlapH = 0.5
        overlapW = 0.5

        interZ = int(resultShape[0] * (1.0 - overlapZ))
        interH = int(resultShape[1] * (1.0 - overlapH))
        interW = int(resultShape[2] * (1.0 - overlapW))

        iterZ = int(((imgShape[0] - resultShape[0]) / interZ) + 1)
        iterH = int(((imgShape[1] - resultShape[1]) / interH) + 1)
        iterW = int(((imgShape[2] - resultShape[2]) / interW) + 1)

        freeZ = imgShape[0] - (resultShape[0] + interZ * (iterZ - 1))
        freeH = imgShape[1] - (resultShape[1] + interH * (iterH - 1))
        freeW = imgShape[2] - (resultShape[2] + interW * (iterW - 1))

        startZ = int(freeZ / 2)
        startH = int(freeH / 2)
        startW = int(freeW / 2)

        total_patches = iterZ*iterH*iterW

        for z in range(0, iterZ):
            for h in range(0, iterH):
                for w in range(0, iterW):
                    input = inputs[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                            (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                            (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))].to(job_description['device'])

                    initial = initials[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                            (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                            (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))].to(job_description['device'])

                    with torch.no_grad():
                        select = 1
                        outputs_rec = initial
                        outputs_seg = model.forward(input, outputs_rec, initials, select)

                    prediction[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += outputs_seg.to('cpu')
                    imgMatrix[ (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                    (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                    (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += resultMatrix

        imgMatrix = np.where(imgMatrix == 0, 1, imgMatrix)
        if job_description['dataset'] == 'vessel':
            labels_np = gt.cpu().detach().numpy()[0, :]
        outputs_np = np.divide(prediction.cpu().detach().numpy()[0, :], imgMatrix)
        outputs_np = np.where(outputs_np > 0.5, 1, 0)

        # outputs_np_ready = np.divide(prediction.cpu().detach().numpy()[0, :], imgMatrix)

        # dice for single class
        if job_description['dataset'] == 'vessel':
            for c in range(0, 1):
                if i == 0:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'w')
                else:
                    file = open(job_description['result_path'] + '/test_cnn' + '/seg_{:d}_list.txt'.format(c),
                                'a')

                dice = 0
                if np.max(labels_np[c]) == 1:
                    dice = Dice_numpy(outputs_np[c], labels_np[c])
                    file.write('{:.4f}\n'.format(dice))
                elif np.max(labels_np[c]) == 0:
                    dice = 0
                    file.write('None\n')
                file.close()
                acc_overall[i, c] = dice

            # dice for all classes
            dice = Dice_numpy(outputs_np, labels_np)
            file_all.write('{:.4f}\n'.format(dice))
            dice_all += dice

        # save files
        outputs_np_ready = unpad_nd_image_3D(outputs_np, job_description['Pad_size'], data_raw.shape[2:]).astype(float)
        for c in range(outputs_np.shape[0]):

            metadata_np = load_pickle(patient + '.pkl')

            original_shape = metadata_np['original_shape']
            seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
            # nonzero = metadata_np['nonzero_region']
            # seg_original_shape[nonzero[0, 0]: nonzero[0, 1] + 1,
            # nonzero[1, 0]: nonzero[1, 1] + 1,
            # nonzero[2, 0]: nonzero[2, 1] + 1] = outputs_np_ready[c]
            seg_original_shape[:] = outputs_np_ready[c]
            # np.save(join(job_description['result_path'] + '/test_cnn/npy/', patient.split(sep='/')[-1] + '_{:s}.npy').format(str(c)), outputs_np_ready[c])
            sitk_output = sitk.GetImageFromArray(seg_original_shape)
            sitk_output.SetDirection(metadata_np['direction'])
            sitk_output.SetOrigin(metadata_np['origin'])
            sitk_output.SetSpacing(tuple(metadata_np['spacing'][[2, 1, 0]]))
            sitk.WriteImage(sitk_output,
                            job_description['result_path'] + '/test_cnn/nii/' + patient.split(sep='/')[-1] + '_{:s}.nii.gz'.format(str(c)))

    for i in range(0, 1):
        acc = np.sum(acc_overall[:, i]) / len(job_description['test_keys'])
        print('seg {:d}: {:.4f}'.format(i, acc))
        acc_avg += acc
        file_acc.write('{:.4f}\n'.format(acc))

    acc_avg = dice_all / len(job_description['test_keys'])
    file_acc.write('\n{:.4f}\n'.format(acc_avg))
    file_acc.close()

    file_all.close()
    print('seg all: {:.4f}'.format(acc_avg))

    print('-' * 64)

    return None


def testing_gan_3D(job_description):
    model = GAN_3D(job_description=job_description).to(job_description['device'])

    mkdir.mkdir(job_description['result_path'] + '/test_cnn/nii')

    model.eval()
    model.load_state_dict(torch.load(job_description['result_path'] + '/model/G_99.pth', map_location=torch.device(job_description['device'])))

    image = sitk.ReadImage("/PythonCodes/label_refinement_3D_GAN/.nii.gz")

    data_image = sitk.GetArrayFromImage(image)

    data = np.expand_dims(np.expand_dims(data_image, axis=0), axis=0)

    # data = pad_nd_image(data_raw, job_description['Pad_size'])

    if job_description['dataset'] == 'airway':
        inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
        labels = torch.from_numpy(data[:, 0:1]).to(job_description['device'], dtype=torch.long)
        # lungs = torch.from_numpy(data[:, 2:3]).to(job_description['device'], dtype=torch.long)
    if job_description['dataset'] == 'vessel':
        inputs = torch.from_numpy(data[:, 0:1]).to(job_description['device'])
        # labels = torch.from_numpy(data[:, 1:2]).to(job_description['device'], dtype=torch.float)
        # labels = one_hot.one_hot_embedding(labels, num_classes=2)
        # labels = labels.permute(0, 4, 1, 2, 3).to(job_description['device'], dtype=torch.float)[:, 1:2]

    # create prediction tensor
    prediction = torch.zeros(data.shape)
    imgShape = inputs.size()[2:]
    resultShape = job_description['Patch_size']

    imgMatrix = np.zeros([imgShape[0], imgShape[1], imgShape[2]], dtype=np.float32)
    resultMatrix = np.ones([resultShape[0], resultShape[1], resultShape[2]], dtype=np.float32)

    overlapZ = 0.5
    overlapH = 0.5
    overlapW = 0.5

    interZ = int(resultShape[0] * (1.0 - overlapZ))
    interH = int(resultShape[1] * (1.0 - overlapH))
    interW = int(resultShape[2] * (1.0 - overlapW))

    iterZ = int(((imgShape[0] - resultShape[0]) / interZ) + 1)
    iterH = int(((imgShape[1] - resultShape[1]) / interH) + 1)
    iterW = int(((imgShape[2] - resultShape[2]) / interW) + 1)

    freeZ = imgShape[0] - (resultShape[0] + interZ * (iterZ - 1))
    freeH = imgShape[1] - (resultShape[1] + interH * (iterH - 1))
    freeW = imgShape[2] - (resultShape[2] + interW * (iterW - 1))

    startZ = int(freeZ / 2)
    startH = int(freeH / 2)
    startW = int(freeW / 2)

    total_patches = iterZ * iterH * iterW

    for z in range(0, iterZ):
        for h in range(0, iterH):
            for w in range(0, iterW):
                input = labels[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                        (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                        (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))].to(
                    job_description['device'], dtype=torch.float)

                with torch.no_grad():
                    outputs_rec, cla, gt_bool = model.forward(input, input, input, "G", 0,
                                                              job_description)

                prediction[:, :, (startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += outputs_rec.to('cpu')
                imgMatrix[(startZ + (interZ * z)):(startZ + (interZ * (z) + resultShape[0])),
                (startH + (interH * h)):(startH + (interH * (h) + resultShape[1])),
                (startW + (interW * w)):(startW + (interW * (w) + resultShape[2]))] += resultMatrix

    imgMatrix = np.where(imgMatrix == 0, 1, imgMatrix)
    outputs_np = np.divide(prediction.cpu().detach().numpy()[0, 0], imgMatrix)
    outputs_np = np.where(outputs_np > 0.5, 1, 0)

    # save files

    sitk_output = sitk.GetImageFromArray(outputs_np.astype(np.uint8))
    sitk.WriteImage(sitk_output,
                    job_description['result_path'] + '/test_cnn/nii/' + 'GAN.nii.gz')


    print('-' * 64)

    return None



