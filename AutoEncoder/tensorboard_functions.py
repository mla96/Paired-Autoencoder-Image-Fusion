import os
from matplotlib import pyplot as plt
plt.switch_backend('agg')


def plot_tensors_tensorboard(input, output, step, epoch, loss, writer, output_path):
    # Create plot path directory if it doesn't exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    fig = plt.figure()
    plt.title("Step " + str(step) + "; Epoch: {}".format(epoch + 1) + "\n Loss: {:.5f}".format(loss), y=0.9)
    plt.axis('off')
    fig.add_subplot(1, 2, 1)
    plt.imshow(input)
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(output)
    plt.axis('off')
    plt.savefig(os.path.join(output_path, "trainfig_step" + str(step)))
    writer.add_figure('train_fig', fig, step)
    print('\n Figure added \n')