import numpy as np


class Viz_visdom():
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
