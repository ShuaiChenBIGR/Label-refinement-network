from src.py.utils.loss import *
import src.py.utils.one_hot_embedding as one_hot
import matplotlib.pyplot as plt
from src.py.utils.visualize_batch import visualize_main_seg, visualize_LR


class eval_val_seg_3D():
    def __init__(self, job_description):
        self.job_description = job_description

    def eval_val(self, model, val_gen, num_validation_batches_per_epoch, epoch, job_description):

        model.eval()
        epoch_val = 0

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)

            if job_description['dataset'] == 'airway':
                inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 0:1]
                lungs = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 1:2]
            else:
                inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)

            labels = one_hot.one_hot_embedding(labels, num_classes=2)
            labels = labels[:, 0].permute(0, 4, 1, 2, 3).to(job_description['device'], dtype=torch.float)

            # create prediction tensor
            with torch.no_grad():
                outputs_seg = model(inputs)

            if job_description['dataset'] == 'airway':
                outputs_seg = outputs_seg * lungs

            outputs_seg_threshold = np.where(outputs_seg.cpu().detach().numpy() > 0.5, 1, 0)
            loss = Dice_numpy(outputs_seg_threshold[:, :], labels.cpu().detach().numpy()[:, 1:])
            epoch_val += loss.item()

            # visualize training set at the end of each epoch
            if job_description['preview_val'] == True:
                if b == 0:
                    slice = int(job_description['Patch_size'][0] / 2)
                    inputs_vis = inputs.cpu().detach().numpy()[0][0][slice]
                    label_vis = labels.cpu().detach().numpy()[0][1][slice]
                    outputs_vis = outputs_seg.cpu().detach().numpy()[0][0][slice]
                    fig_batch = plt.figure(figsize=[7, 10])

                    visualize_main_seg(inputs_vis, label_vis, outputs_vis, epoch=epoch)

                    plt.savefig(job_description['result_path'] + '/preview_main/val/' + 'epoch_%s.jpg' % epoch)
                    plt.close(fig_batch)

        return epoch_val / num_validation_batches_per_epoch


class eval_val_adv_3D():
    def __init__(self, job_description):
        self.job_description = job_description

    def eval_val(self, model, val_gen, num_validation_batches_per_epoch, epoch, job_description):

        model.eval()
        epoch_val = 0

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)

            if job_description['dataset'] == 'airway':
                inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 0:1]
                lungs = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 1:2]
            else:
                inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)

            labels = one_hot.one_hot_embedding(labels, num_classes=2)
            labels = labels[:, 0].permute(0, 4, 1, 2, 3).to(job_description['device'], dtype=torch.float)

            # create prediction tensor
            with torch.no_grad():
                cla, selection, outputs_seg = model(inputs, labels[:, 1:2])

            if job_description['dataset'] == 'airway':
                outputs_seg = outputs_seg * lungs

            outputs_seg_threshold = np.where(outputs_seg.cpu().detach().numpy() > 0.5, 1, 0)

            loss = Dice_numpy(outputs_seg_threshold[:, :], labels.cpu().detach().numpy()[:, 1:])
            epoch_val += loss.item()

            # visualize training set at the end of each epoch
            if job_description['preview_val'] == True:
                if b == 0:
                    slice = int(job_description['Patch_size'][0] / 2)
                    inputs_vis = inputs.cpu().detach().numpy()[0][0][slice]
                    label_vis = labels.cpu().detach().numpy()[0][1][slice]
                    outputs_vis = outputs_seg.cpu().detach().numpy()[0][0][slice]
                    fig_batch = plt.figure(figsize=[7, 10])

                    visualize_main_seg(inputs_vis, label_vis, outputs_vis, epoch=epoch)

                    plt.savefig(job_description['result_path'] + '/preview_main/val/' + 'epoch_%s.jpg' % epoch)
                    plt.close(fig_batch)

        return epoch_val / num_validation_batches_per_epoch


class eval_val_LR_3D():
    def __init__(self, job_description):
        self.job_description = job_description

    def eval_val(self, model, val_gen, num_validation_batches_per_epoch, epoch, job_description):

        model.eval()
        epoch_val = 0

        for b in range(num_validation_batches_per_epoch):
            batch = next(val_gen)

            if job_description['dataset'] == 'airway':
                inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 0:1]
                lungs = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 1:2]
                initials = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 2:3]
            else:
                inputs = torch.from_numpy(batch['data']).to(job_description['device'])
                labels = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 0:1]
                initials = torch.from_numpy(batch['seg']).to(job_description['device'], dtype=torch.long)[:, 1:2]

            labels = one_hot.one_hot_embedding(labels, num_classes=2)
            labels = labels[:, 0].permute(0, 4, 1, 2, 3).to(job_description['device'], dtype=torch.float)

            # create prediction tensor
            with torch.no_grad():
                select = 1
                outputs_rec = initials
                outputs_seg = model.forward(inputs, outputs_rec, initials, select)

            if job_description['dataset'] == 'airway':
                outputs_seg = outputs_seg * lungs

            outputs_seg_threshold = np.where(outputs_seg.cpu().detach().numpy() > 0.5, 1, 0)
            loss = Dice_numpy(outputs_seg_threshold[:, :], labels.cpu().detach().numpy()[:, 1:])
            epoch_val += loss.item()

            # visualize training set at the end of each epoch
            if job_description['preview_val'] == True:
                if b == 0:
                    slice = int(job_description['Patch_size'][0] / 2)
                    errors_vis = initials.cpu().detach().numpy()[0][1][slice]
                    inputs_vis = outputs_rec.cpu().detach().numpy()[0][0][slice]
                    initials_vis = initials.cpu().detach().numpy()[0][0][slice]
                        
                    label_vis = labels.cpu().detach().numpy()[0][1][slice]
                    outputs_vis = outputs_seg.cpu().detach().numpy()[0][0][slice]

                    fig_batch = plt.figure(figsize=[7, 10])
                    visualize_LR(errors_vis, inputs_vis, initials_vis, label_vis, outputs_vis, epoch=epoch)

                    plt.savefig(job_description['result_path'] + '/preview_main/val/' + 'epoch_%s.jpg' % epoch)
                    plt.close(fig_batch)

        return epoch_val / num_validation_batches_per_epoch
