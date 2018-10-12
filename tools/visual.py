import torch
import numpy as np
import matplotlib.pyplot as plt


class Viz_visdom(object):
    def __init__(self, name, display_id=0):
        self.name = name
        self.display_id = display_id
        self.idx = display_id
        self.plot_data = {}
        if display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=8097)

    def plot_current_errors(self, epoch, counter_ratio, errors, idx=0):

        if idx not in self.plot_data:
            self.plot_data[idx] = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        # self.plot_data = {'X': [], 'Y': [], 'legend': list(errors.keys())}
        self.plot_data[idx]['X'].append(epoch + counter_ratio)
        self.plot_data[idx]['Y'].append([errors[k] for k in self.plot_data[idx]['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data[idx]['X'])] * len(self.plot_data[idx]['legend']), 1)
            if len(errors) > 1 else np.array(self.plot_data[idx]['X']),
            Y=np.array(self.plot_data[idx]['Y']) if len(errors) > 1 else np.array(self.plot_data[idx]['Y'])[:, 0],
            opts={
                'title': self.name + ' loss over time %d' % idx,
                'legend': self.plot_data[idx]['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id + idx)
        if self.idx < self.display_id + idx:
            self.idx = self.display_id + idx

    def plot_current_img(self, visuals, c_prev=True):
        idx = self.idx + 1
        for label, image_numpy in visuals.items():
            if c_prev:
                self.vis.image(image_numpy, opts=dict(title=label),
                               win=self.display_id + idx)
            else:
                image_numpy = image_numpy.swapaxes(0, 2).swapaxes(1, 2)
                self.vis.image(image_numpy, opts=dict(title=label),
                               win=self.display_id + idx)
            idx += 1


# reference: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
def plot_image(inp, fig_size, title=None, swap_channel=False, norm=False):
    """Imshow for Tensor."""
    if torch.is_tensor(inp):
        inp = inp.numpy().transpose((1, 2, 0)) if swap_channel else inp.numpy()
    else:
        inp = inp.transpose((1, 2, 0)) if swap_channel else inp
    if norm:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=fig_size)
    if inp.shape[0] == 1:
        plt.imshow(inp[0], cmap='gray')
    else:
        plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)  # pause a bit so that plots are updated


def make_simple_grid(inp, padding=2, padding_value=1):
    inp = torch.stack(inp, dim=0)
    nmaps = inp.size(0)
    height, width = inp.size(2), int(inp.size(3) + padding)
    grid = inp.new(1, height, width * nmaps + padding).fill_(padding_value)
    for i in range(nmaps):
        grid.narrow(2, i * width + padding, width - padding).copy_(inp[i])
    return grid


if __name__ == '__main__':
    inp = [torch.randn(1, 5, 5), torch.randn(1, 5, 5)]
    out = make_simple_grid(inp)
    print(out.size())
    plot_image(out)
