from matplotlib import pyplot as plt
plt.switch_backend('agg')


def plot_tensors_tensorboard(input, output, step, writer):
    fig = plt.figure()
    plt.title("Step " + str(step), y=0.9)
    plt.axis('off')
    fig.add_subplot(1, 2, 1)
    plt.imshow(input)
    plt.axis('off')
    fig.add_subplot(1, 2, 2)
    plt.imshow(output)
    plt.axis('off')
    writer.add_figure('train_fig', fig, step)
    print('\n Figure added \n')