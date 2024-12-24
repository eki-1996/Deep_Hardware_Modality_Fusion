from matplotlib import animation, pyplot as plt


def save_gif(frames, file_path, vmax=255, vmin=0, interval=3000/25):
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ims = []
        plt.xticks([])
        plt.yticks([])
        plt.grid(True)
        for frame in frames:
            m = plt.imshow(
                (frame).reshape(*frame.shape[:-1]),
                cmap=plt.cm.gray, vmax=vmax, vmin=vmin)
            plt.axis('off')
            ims.append([m])
        ani = animation.ArtistAnimation(fig, ims, interval=interval, repeat=False)
        ani.save(file_path, writer="imagemagick")
        plt.close()