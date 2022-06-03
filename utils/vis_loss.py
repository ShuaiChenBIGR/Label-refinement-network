import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Visualize training loss:
def visualize_loss(fig, row=1, col=2, dict=None, title=None, epoch=None):
    if epoch is not None:
        fig.suptitle('epoch: {}'.format(epoch), fontsize=16)

    start_epoch = 0

    ax1 = plt.subplot(row, col, 1)
    # plt.ylim(0, 1)
    ax1.set_title('training loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('CNN')
    lns1 = plt.plot(dict[title[0]][start_epoch:], dict[title[1]][start_epoch:], 'royalblue', label='CNN')
    plt.grid(which='major', axis='y', linestyle='--')

    lns = lns1
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    ax1 = plt.subplot(row, col, 2)
    # plt.ylim(0, 1)
    ax1.set_title('val performance')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('CNN')
    lns1 = plt.plot(dict[title[0]][start_epoch:], dict[title[2]][start_epoch:], 'royalblue', label='CNN')

    plt.grid(which='major', axis='y', linestyle='--')

    # ax3 = ax1.twinx()
    # # plt.ylim(0, 1)
    # ax3.set_xlabel('epoch')
    # ax3.set_ylabel('MSE')
    # lns3 = plt.plot(dict[title[0]][20:], dict[title[6]][20:], 'green', label='KL')
    lns = lns1
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig


def visualize_GAN_loss(fig, row=1, col=4, dict=None, title=None, epoch=None):
    if epoch is not None:
        fig.suptitle('epoch: {}'.format(epoch), fontsize=16)

    start_epoch = 0

    ax1 = plt.subplot(row, col, 1)
    # plt.ylim(0, 1)
    ax1.set_title('training G_initial')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('G_initial')
    lns1 = plt.plot(dict[title[0]][start_epoch:], dict[title[1]][start_epoch:], 'royalblue', label='CNN')
    plt.grid(which='major', axis='y', linestyle='--')

    lns = lns1
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    ax1 = plt.subplot(row, col, 2)
    # plt.ylim(0, 1)
    ax1.set_title('training G_error')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('G_error')
    lns1 = plt.plot(dict[title[0]][start_epoch:], dict[title[2]][start_epoch:], 'royalblue', label='CNN')
    plt.grid(which='major', axis='y', linestyle='--')

    lns = lns1
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    ax1 = plt.subplot(row, col, 3)
    # plt.ylim(0, 1)
    ax1.set_title('training Adv')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Adv')
    lns1 = plt.plot(dict[title[0]][start_epoch:], dict[title[3]][start_epoch:], 'royalblue', label='CNN')

    plt.grid(which='major', axis='y', linestyle='--')

    # ax3 = ax1.twinx()
    # # plt.ylim(0, 1)
    # ax3.set_xlabel('epoch')
    # ax3.set_ylabel('MSE')
    # lns3 = plt.plot(dict[title[0]][20:], dict[title[6]][20:], 'green', label='KL')
    lns = lns1
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    ax1 = plt.subplot(row, col, 4)
    # plt.ylim(0, 1)
    ax1.set_title('training D')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Discriminator')
    lns1 = plt.plot(dict[title[0]][start_epoch:], dict[title[4]][start_epoch:], 'royalblue', label='CNN')

    plt.grid(which='major', axis='y', linestyle='--')

    # ax3 = ax1.twinx()
    # # plt.ylim(0, 1)
    # ax3.set_xlabel('epoch')
    # ax3.set_ylabel('MSE')
    # lns3 = plt.plot(dict[title[0]][20:], dict[title[6]][20:], 'green', label='KL')
    lns = lns1
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    return fig


def history_log(path, history, write_stat):
    # print(history)
    the_file = open(path, write_stat)
    the_file.write(history)
    the_file.close()