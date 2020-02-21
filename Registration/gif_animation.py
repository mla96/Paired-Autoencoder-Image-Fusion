# Michelle La
# Jan. 30, 2020


import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Input a 3D time series np array and convert to B/W movie, and save it as filename
def voxel_movie(array, filename):
    array = [array[:, :, i] for i in range(1024)]

    fig = plt.figure()
    im = plt.imshow(array[512], cmap='gray')

    def update_voxel_movie(j):
        im.set_array(array[j])
        return im

    ani = animation.FuncAnimation(fig, update_voxel_movie, frames=1024, interval=5)
    # plt.show()
    ani.save(filename + '.gif', writer='imagemagick', fps=5, bitrate=5)

    return ani
