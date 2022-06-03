from src.py.training_on_the_fly import train_CNN_3D, train_Adv_3D, train_Dou_3D, train_GAN_3D, train_LR_3D, train_LR_Syn_3D, train_LR_Syn_GAN_3D, train_LR_Syn_GAN_3D_semi, testing_3D
import src.py.utils.info as info


def call_job_3D(job_description):

    print('-' * 64)
    print('job starts')
    print('-' * 64)

    print('current dataset: ', job_description['dataset'])
    print('method: ', job_description['method'])
    print('random seed: ', job_description['seed'])
    print('network: ', job_description['network'])
    print('batch size: ', job_description['batch_size'])
    print('-' * 64)

    info.history_log(job_description['result_path'] + '/info.txt', '-' * 64 + '\n', 'w')
    info.history_log(job_description['result_path'] + '/info.txt', 'job starts' + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt', '-' * 64 + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt', 'dataset: ' + job_description['dataset'] + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'base features: ' + str(job_description['base_features']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'learning rate: ' + str(job_description['lr']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'random seed: ' + str(job_description['seed']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'batch size: ' + str(job_description['batch_size']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'Patch_size: ' + str(job_description['Patch_size']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'Pad_size: ' + str(job_description['Pad_size']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'lower_bound: ' + str(job_description['error_lower_bound']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'upper_bound: ' + str(job_description['error_upper_bound']) + '\n', 'a')
    info.history_log(job_description['result_path'] + '/info.txt',
                     'error_size: ' + str(job_description['error_size']) + '\n', 'a')


    # Choose method
    if job_description['mode'] == 'train' and job_description['method'] == 'CNN':
        train_CNN_3D.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'CNN':
        testing_3D.testing_seg_3D(job_description)

    if job_description['mode'] == 'train' and job_description['method'] == 'Adv':
        train_Adv_3D.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'Adv':
        testing_3D.testing_adv_3D(job_description)

    if job_description['mode'] == 'train' and job_description['method'] == 'DoubleUNet':
        train_Dou_3D.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'DoubleUNet':
        testing_3D.testing_dou_3D(job_description)

    if job_description['mode'] == 'train' and job_description['method'] == 'GAN':
        train_GAN_3D.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'GAN':
        testing_3D.testing_gan_3D(job_description)

    if job_description['mode'] == 'train' and job_description['method'] == 'LR':
        train_LR_3D.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'LR':
        testing_3D.testing_lr_3D(job_description)

    if job_description['mode'] == 'train' and job_description['method'] == 'LR_Syn':
        train_LR_Syn_3D.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'LR_Syn':
        testing_3D.testing_lr_3D(job_description)

    if job_description['mode'] == 'train' and job_description['method'] == 'LR_Syn_GAN':
        train_LR_Syn_GAN_3D.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'LR_Syn_GAN':
        testing_3D.testing_lr_3D(job_description)

    if job_description['mode'] == 'train' and job_description['method'] == 'LR_Syn_GAN_semi':
        train_LR_Syn_GAN_3D_semi.start_main_3D(job_description)
    if job_description['mode'] == 'test' and job_description['method'] == 'LR_Syn_GAN_semi':
        testing_3D.testing_lr_3D(job_description)

    return None
