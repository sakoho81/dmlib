#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import itertools
from skimage import draw
import numpy as np

from PyQt5.QtWidgets import QInputDialog


class DMPlot:

    def __init__(self, geometry='round', sampling=128, nact=12, pitch=.3, roll=2, mapmul=.3, txs=(0, 0, 0)):
        self.floor = -1.5
        assert geometry in ('round', 'square')
        self.geometry = geometry
        if geometry == 'round':
            self.nact_x_diam = int(round(2 * np.sqrt(nact / np.pi)))
        else:
            self.nact_x_diam = int(round(np.sqrt(nact)))

        self.sampling = sampling
        self.pitch = pitch
        self.roll = roll

        self.mapmul = mapmul
        self.txs = txs

        self._make_grids()

    def update_txs(self, txs):
        self.txs[:] = txs[:]
        self._make_grids()

    def flipx(self, b):
        self.txs[0] = b
        self._make_grids()

    def flipy(self, b):
        self.txs[1] = b
        self._make_grids()

    def rotate(self, p):
        self.txs[2] = p
        self._make_grids()

    def _make_grids(self):

        d = np.linspace(-1, 1, self.nact_x_diam)
        d *= self.pitch/np.diff(d)[0]
        x, y = np.meshgrid(d, d)
        if self.txs[2]:
            x = np.rot90(x, self.txs[2])
            y = np.rot90(y, self.txs[2])
        if self.txs[0]:
            if self.txs[2] % 2:
                x = np.flipud(x)
            else:
                x = np.fliplr(x)
        if self.txs[1]:
            if self.txs[2] % 2:
                y = np.fliplr(y)
            else:
                y = np.flipud(y)

        dd = np.linspace(d.min() - self.pitch, d.max() + self.pitch, self.sampling)
        xx, yy = np.meshgrid(dd, dd)

        maps = []
        acts = []
        index = []
        if self.geometry == 'square':
            exclude = list(itertools.product((0, self.nact_x_diam - 1), repeat=2))
        elif self.geometry == 'round':
            circle = np.zeros((self.nact_x_diam,) * 2)
            center = list(length//2 for length in circle.shape)
            circle[draw.circle(*center, self.nact_x_diam / 2, shape=circle.shape)] = 1
            exclude = np.column_stack(np.where(circle == 0)).tolist()
        else:
            raise AttributeError("Invalid DM shape")

        #exclude = [(0, 0), (0, 11), (11, 0), (11, 11)]
        count = 1
        patvis = []
        for i in range(x.shape[1]):



            for j in range(y.shape[0]):
                if [i, j] in exclude:
                    continue

                r = np.sqrt((xx - x[i, j])**2 + (yy - y[i, j])**2)
                z = np.exp(-self.roll*r/self.pitch)
                acts.append(z.reshape(-1, 1))

                mp = np.logical_and(
                    np.abs(xx - x[i, j]) < self.mapmul*self.pitch,
                    np.abs(yy - y[i, j]) < self.mapmul*self.pitch)
                maps.append(mp)
                index.append(count*mp.reshape(-1, 1))
                patvis.append(mp.reshape(-1, 1).astype(np.float))
                count += 1

        self.A_shape = xx.shape
        self.A = np.hstack(acts)
        self.maps = maps
        self.layout = np.sum(np.dstack(maps), axis=2)
        self.pattern = np.hstack(index)
        self.index = np.sum(np.hstack(index), axis=1)
        self.patvis = np.hstack(patvis)
        self.mappatvis = np.invert(self.layout.astype(np.bool)).ravel()

    def size(self):
        return self.A.shape[1]

    def compute_gauss(self, u):
        pat = np.dot(self.A, u)
        return pat.reshape(self.A_shape)

    def compute_pattern(self, u):
        pat = np.dot(self.patvis, u)
        pat[self.mappatvis] = self.floor
        return pat.reshape(self.A_shape)

    def index_actuator(self, x, y):
        return self.index[int(y)*self.sampling + int(x)] - 1

    def install_select_callback(self, ax, u, parent, write=None):
        def f(e):
            if e.inaxes is not None:
                ind = self.index_actuator(e.xdata, e.ydata)
                if ind != -1:
                    val, ok = QInputDialog.getDouble(
                        parent, 'Actuator ' + str(ind), 'range [-1, 1]',
                        u[ind], -1., 1., 4)
                    if ok:
                        u[ind] = val
                        self.draw(ax, u)
                        ax.figure.canvas.draw()
                        if write:
                            write(u)

        ax.figure.canvas.callbacks.connect('button_press_event', f)

    def draw(self, ax, u):
        return ax.imshow(self.compute_pattern(u), vmin=self.floor, vmax=1)
