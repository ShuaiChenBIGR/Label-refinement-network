import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def visualize_main_seg(inputs_vis, label_vis, outputs_vis, row=3, col=1, epoch = None):

    if epoch is not None:
        plt.suptitle('epoch: {}'.format(epoch), fontsize=16)

    ax = plt.subplot(row, col, 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Image', fontsize=12)
    ax.axis('off')
    plt.imshow(inputs_vis, cmap='gray', alpha=1.0)

    ax = plt.subplot(row, col, col + 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('ground truth', fontsize=12)
    ax.axis('off')
    plt.imshow(label_vis, cmap='viridis', alpha=1.0)

    ax = plt.subplot(row, col, col*2 + 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('prediction', fontsize=12)
    ax.axis('off')
    plt.imshow(outputs_vis, cmap='viridis', alpha=1.0)

    return None


def visualize_LR(errors_vis, inputs_vis, initials_vis, label_vis, outputs_vis, row=2, col=3, epoch = None):

    if epoch is not None:
        plt.suptitle('epoch: {}'.format(epoch), fontsize=16)

    ax = plt.subplot(row, col, 1)
    ax.set_title('errors', fontsize=12)
    ax.axis('off')
    plt.imshow(errors_vis, cmap='gray', alpha=1.0)

    ax = plt.subplot(row, col, 2)
    ax.set_title('errors with GAN', fontsize=12)
    ax.axis('off')
    plt.imshow(inputs_vis, cmap='gray', alpha=1.0)

    ax = plt.subplot(row, col, 3)
    ax.set_title('initials', fontsize=12)
    ax.axis('off')
    plt.imshow(initials_vis, cmap='gray', alpha=1.0)

    ax = plt.subplot(row, col, 4)
    ax.set_title('ground truth', fontsize=12)
    ax.axis('off')
    plt.imshow(label_vis, cmap='viridis', alpha=1.0)

    ax = plt.subplot(row, col, 5)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('prediction', fontsize=12)
    ax.axis('off')
    plt.imshow(outputs_vis, cmap='viridis', alpha=1.0)

    return None


def visualize_GAN(labels_original_vis, inputs_vis, label_vis, outputs_vis, row=2, col=2, epoch=None, ratio=0):

    if epoch is not None:
        plt.suptitle('epoch: {}, ratio: {}'.format(epoch, ratio), fontsize=16)

    ax = plt.subplot(row, col, 1)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Original GT', fontsize=12)
    ax.axis('off')
    plt.imshow(labels_original_vis, cmap='viridis', alpha=1.0, vmin=0, vmax=1)

    ax = plt.subplot(row, col, 2)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Generated Errors', fontsize=12)
    ax.axis('off')
    plt.imshow(inputs_vis, cmap='viridis', alpha=1.0, vmin=0, vmax=1)

    ax = plt.subplot(row, col, 3)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Initials', fontsize=12)
    ax.axis('off')
    plt.imshow(label_vis, cmap='viridis', alpha=1.0, vmin=0, vmax=1)

    ax = plt.subplot(row, col, 4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Outputs', fontsize=12)
    ax.axis('off')
    plt.imshow(outputs_vis, cmap='viridis', alpha=1.0, vmin=0, vmax=1)

    return None

