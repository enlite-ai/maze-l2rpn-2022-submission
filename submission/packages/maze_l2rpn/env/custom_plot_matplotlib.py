# Adopted from the original power grid simulation with the following copyright notice
# Copyright (c) 2019-2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of Grid2Op, Grid2Op a testbed platform to model sequential decision making in power systems.

import io

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Observation import CompleteObservation
from grid2op.PlotGrid.BasePlot import BasePlot
from grid2op.PlotGrid.PlotUtil import PlotUtil as pltu
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.path import Path


class CustomPlotMatplot(BasePlot):
    """
    Attributes
    ----------

    width: ``int``
        Width of the figure in pixels
    height: ``int``
        Height of the figure in pixel
    dpi: ``int``
        Dots per inch, to convert pixels dimensions into inches
    _scale: ``float``
        Scale of the drawing in arbitrary units
    _sub_radius: ``int``
        Substation circle size
    _sub_face_color: ``str``
        Substation circle fill color
    _sub_edge_color: ``str``
        Substation circle edge color
    _sub_txt_color: ``str``
        Substation info text color
    _load_radius: ``int``
        Load circle size
    _load_name: ``bool``
        Show load names (default True)
    _load_face_color: ``str``
        Load circle fill color
    _load_edge_color: ``str``
        Load circle edge color
    _load_txt_color: ``str``
        Load info text color
    _load_line_color: ``str``
        Color of the line from load to substation
    _load_line_width: ``int``
        Width of the line from load to substation
    _gen_radius: ``int``
        Generators circle size
    _gen_name: ``bool``
        Show generators names (default True)
    _gen_face_color: ``str``
        Generators circle fill color
    _gen_edge_color: ``str``
        Generators circle edge color
    _gen_txt_color: ``str``
        Generators info txt color
    _gen_line_color: ``str``
        Color of the line form generator to substation
    _gen_line_width: ``str``
        Width of the line from generator to substation
    _line_color_scheme: ``list``
        List of color strings to color powerlines based on rho values
    _line_color_width: ``int``
        Width of the powerlines lines
    _line_bus_radius: ``int``
        Size of the bus display circle
    _line_bus_face_colors: ``list``
        List of 3 colors strings, each corresponding to the fill color of the bus circle
    _line_arrow_len: ``int``
        Length of the arrow on the powerlines
    _line_arrow_width: ``int``
       Width of the arrow on the powerlines
    """

    def draw_storage(self, figure, observation, storage_name, storage_id, storage_bus, storage_value, storage_unit,
                     pos_x, pos_y, sub_x, sub_y):
        # TODO
        raise NotImplementedError

    def __init__(self,
                 observation_space,
                 width=1600,
                 height=900,
                 grid_layout=None,
                 dpi=96,
                 scale=2000.0,
                 sub_radius=15,
                 load_radius=8,
                 load_name=False,
                 load_id=False,
                 gen_radius=8,
                 gen_name=False,
                 gen_id=False,
                 line_name=False,
                 line_id=False):
        self.dpi = dpi
        super().__init__(observation_space, width, height, scale, grid_layout)

        self._sub_radius = sub_radius
        self._sub_face_color = "w"
        self._sub_edge_color = "blue"
        self._sub_txt_color = "black"

        self._load_radius = load_radius
        self._load_name = load_name
        self._load_id = load_id
        self._load_face_color = "w"
        self._load_edge_color = "orange"
        self._load_txt_color = "black"
        self._load_line_color = "black"
        self._load_line_width = 1

        self._gen_radius = gen_radius
        self._gen_name = gen_name
        self._gen_id = gen_id
        self._gen_face_color = "w"
        self._gen_edge_color = "green"
        self._gen_txt_color = "black"
        self._gen_line_color = "black"
        self._gen_line_width = 1

        # cx = np.linspace(0.0, 0.70, 10)
        # self._line_color_scheme = cm.get_cmap("inferno")(cx)
        self._line_name = line_name
        self._line_id = line_id
        self._line_color_scheme = ["blue", "orange", "red"]
        self._line_color_width = 1
        self._line_bus_radius = 6
        self._line_bus_face_colors = ["black", "red", "lime"]
        self._line_arrow_len = 10
        self._line_arrow_width = 10.0

        self.xlim = [0, 0]
        self.xpad = 5
        self.ylim = [0, 0]
        self.ypad = 5

        self.max_rho = []
        self.history = 50
        self.topo = []
        self.active_lines = []
        self.overflown_lines = []
        self.old_topo = None
        self.step_counter = 0

        self.line_coordinates = {}
        self.load_coordinates = {}
        self.gen_coordinates = {}

        # Assemble mapping from topology vector to entity
        self.map_topo_to_entity = {}
        for _, topo_idx, in enumerate(observation_space.gen_pos_topo_vect):
            self.map_topo_to_entity[topo_idx] = 'generator'
        for _, topo_idx, in enumerate(observation_space.line_ex_pos_topo_vect):
            self.map_topo_to_entity[topo_idx] = 'line_ex'
        for _, topo_idx, in enumerate(observation_space.line_or_pos_topo_vect):
            self.map_topo_to_entity[topo_idx] = 'line_or'
        for _, topo_idx, in enumerate(observation_space.load_pos_topo_vect):
            self.map_topo_to_entity[topo_idx] = 'load'

    def reset_values(self):
        self.max_rho = []
        self.history = 50
        self.topo = []
        self.active_lines = []
        self.overflown_lines = []
        self.old_topo = None
        self.step_counter = 0

    def plot_obs(self, observation: CompleteObservation,
                 figure=None,
                 redraw=True,
                 line_info="rho",
                 load_info="p",
                 gen_info="p",
                 max_rhos=None,
                 topos=None,
                 active_lines=None,
                 overflown_lines=None):
        super(CustomPlotMatplot, self).plot_obs(observation, figure, redraw, line_info, load_info, gen_info)

        if max_rhos is None:
            self.max_rho.append(observation.rho.max())
        else:
            self.max_rho = max_rhos
        if topos is None:
            self.topo.append(observation.topo_vect)
        else:
            self.topo = topos
        if active_lines is None:
            self.active_lines.append(np.sum(observation.line_status))
        else:
            self.active_lines = active_lines
        if overflown_lines is None:
            self.overflown_lines.append(len(observation.rho[observation.rho > 1.0]))
        else:
            self.overflown_lines = overflown_lines

        non_equal_indices = None

        if len(self.topo) > 1:
            non_equal_indices = np.where((self.topo[-2] == observation.topo_vect) == 0)

        if not figure:
            figure = plt.gcf()

        # Plot rho
        ax1 = figure.add_subplot(self.gs[0, 3])
        ax1.set_title('Maximum overflow in any line')
        ax1.set_ylim(0.0, 2.0)
        if self.step_counter <= self.history:
            ax1.plot([0, self.history], [1, 1], "m-", alpha=0.5)
            ax1.plot(np.arange(0, len(self.max_rho[-self.history::])), self.max_rho[-self.history::])
            ax1.set_xticks(np.arange(0, self.history + 1, step=5))
        else:
            ax1.plot([self.step_counter - self.history, self.step_counter],
                     [1, 1], "m-", alpha=0.5)
            ax1.plot(np.arange(self.step_counter - len(self.max_rho[-self.history::]), self.step_counter),
                     self.max_rho[-self.history::])
        ax1.yaxis.grid()

        # Plot active lines
        ax2 = figure.add_subplot(self.gs[1, 3])
        ax2.set_title('Number of active power lines')
        ax2.set_ylim(observation.n_line // 2, observation.n_line + 1)
        if self.step_counter <= self.history:
            ax2.plot(np.arange(0, len(self.active_lines[-self.history::])), self.active_lines[-self.history::])
            ax2.set_xticks(np.arange(0, self.history + 1, step=5))
        else:
            ax2.plot(np.arange(self.step_counter - len(self.active_lines[-self.history::]), self.step_counter),
                     self.active_lines[-self.history::])
        ax2.yaxis.grid()

        # Plot overflown lines
        ax3 = figure.add_subplot(self.gs[2, 3])
        ax3.set_title('Number of overflown power lines')
        ax3.set_ylim(-1, observation.n_line // 2)
        if self.step_counter <= self.history:
            ax3.plot(np.arange(0, len(self.overflown_lines[-self.history::])), self.overflown_lines[-self.history::])
            ax3.set_xticks(np.arange(0, self.history + 1, step=5))
        else:
            ax3.plot(np.arange(self.step_counter - len(self.overflown_lines[-self.history::]), self.step_counter),
                     self.overflown_lines[-self.history::])
        ax3.yaxis.grid()

        plt.xlabel('time step')
        plt.tight_layout()

        # Plot markers at positions where topology changed
        if non_equal_indices is not None:
            for idx in non_equal_indices[0]:
                # Get entity to look up
                entity = self.map_topo_to_entity[idx]
                if entity == 'generator':
                    x = self.gen_coordinates[np.where(observation.gen_pos_topo_vect == idx)[0][0]][0]
                    y = self.gen_coordinates[np.where(observation.gen_pos_topo_vect == idx)[0][0]][1]
                elif entity == 'line_or':
                    x = self.line_coordinates[np.where(observation.line_or_pos_topo_vect == idx)[0][0]][0]
                    y = self.line_coordinates[np.where(observation.line_or_pos_topo_vect == idx)[0][0]][1]
                elif entity == 'line_ex':
                    x = self.line_coordinates[np.where(observation.line_ex_pos_topo_vect == idx)[0][0]][2]
                    y = self.line_coordinates[np.where(observation.line_ex_pos_topo_vect == idx)[0][0]][3]
                elif entity == 'load':
                    x = self.load_coordinates[np.where(observation.load_pos_topo_vect == idx)[0][0]][0]
                    y = self.load_coordinates[np.where(observation.load_pos_topo_vect == idx)[0][0]][1]
                else:
                    raise ValueError("Unknown entity!")

                circle = plt.Circle((x, y), radius=10, color='red')
                ax0 = figure.axes[0]
                ax0.add_patch(circle)

        self.step_counter += 1

    def _v_textpos_from_dir(self, dirx, diry):
        if diry > 0:
            return "bottom"
        else:
            return "top"

    def _h_textpos_from_dir(self, dirx, diry):
        if dirx == 0:
            return "center"
        elif dirx > 0:
            return "left"
        else:
            return "right"

    def create_figure(self):
        # lazy loading of graphics library (reduce loading time)
        # and mainly because matplolib has weird impact on argparse
        w_inch = self.width / self.dpi
        h_inch = self.height / self.dpi
        f = plt.figure(figsize=(w_inch, h_inch), dpi=self.dpi)
        self.gs = GridSpec(3, 4, figure=f)
        self.ax = f.add_subplot(self.gs[:, :3])
        f.canvas.draw()
        return f

    def clear_figure(self, figure):
        self.xlim = [0, 0]
        self.ylim = [0, 0]
        figure.clear()
        # self.ax = figure.subplots()

    def convert_figure_to_numpy_HWC(self, figure):
        w, h = figure.canvas.get_width_height()
        buf = io.BytesIO()
        figure.canvas.print_raw(buf)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img_arr = np.reshape(img_arr, (h, w, 4))
        return img_arr

    def _draw_substation_txt(self, pos_x, pos_y, text):
        self.ax.text(pos_x, pos_y, text,
                     color=self._sub_txt_color,
                     horizontalalignment='center',
                     verticalalignment='center')

    def _draw_substation_circle(self, pos_x, pos_y):
        patch = patches.Circle((pos_x, pos_y),
                               radius=self._sub_radius,
                               facecolor=self._sub_face_color,
                               edgecolor=self._sub_edge_color)
        self.ax.add_patch(patch)

    def draw_substation(self, figure, observation,
                        sub_id, sub_name,
                        pos_x, pos_y):
        self.xlim[0] = min(self.xlim[0], pos_x - self._sub_radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self._sub_radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self._sub_radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self._sub_radius)

        self._draw_substation_circle(pos_x, pos_y)
        self._draw_substation_txt(pos_x, pos_y, str(sub_id))

    def _draw_load_txt(self, pos_x, pos_y, sub_x, sub_y, text):
        dir_x, dir_y = pltu.vec_from_points(sub_x, sub_y, pos_x, pos_y)
        off_x, off_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        txt_x = pos_x + off_x * self._gen_radius
        txt_y = pos_y + off_y * self._gen_radius
        ha = self._h_textpos_from_dir(dir_x, dir_y)
        va = self._v_textpos_from_dir(dir_x, dir_y)
        self.ax.text(txt_x, txt_y, text,
                     color=self._load_txt_color,
                     horizontalalignment=ha,
                     fontsize='small',
                     verticalalignment=va)

    def _draw_load_name(self, pos_x, pos_y, txt):
        self.ax.text(pos_x, pos_y, txt,
                     color=self._load_txt_color,
                     va='center', ha='center',
                     fontsize='x-small')

    def _draw_load_circle(self, pos_x, pos_y):
        patch = patches.Circle((pos_x, pos_y),
                               radius=self._load_radius,
                               facecolor=self._load_face_color,
                               edgecolor=self._load_edge_color)
        self.ax.add_patch(patch)

    def _draw_load_line(self, pos_x, pos_y, sub_x, sub_y):
        codes = [
            Path.MOVETO,
            Path.LINETO
        ]
        verts = [
            (pos_x, pos_y),
            (sub_x, sub_y)
        ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path,
                                  color=self._load_line_color,
                                  lw=self._load_line_width)
        self.ax.add_patch(patch)

    def _draw_load_bus(self,
                       pos_x, pos_y,
                       norm_dir_x, norm_dir_y,
                       bus_id):
        center_x = pos_x + norm_dir_x * self._sub_radius
        center_y = pos_y + norm_dir_y * self._sub_radius
        face_color = self._line_bus_face_colors[bus_id]
        patch = patches.Circle((center_x, center_y),
                               radius=self._line_bus_radius,
                               facecolor=face_color)
        self.ax.add_patch(patch)

    def draw_load(self, figure, observation,
                  load_id, load_name, load_bus,
                  load_value, load_unit,
                  pos_x, pos_y,
                  sub_x, sub_y):

        self.load_coordinates[load_id] = (pos_x, pos_y)

        self.xlim[0] = min(self.xlim[0], pos_x - self._load_radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self._load_radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self._load_radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self._load_radius)
        self._draw_load_line(pos_x, pos_y, sub_x, sub_y)
        self._draw_load_circle(pos_x, pos_y)
        load_txt = ""
        if self._load_name:
            load_txt += "\"{}\":\n".format(load_name)
        if self._load_id:
            load_txt += "id: {}\n".format(load_id)
        if load_value is not None:
            load_txt += pltu.format_value_unit(load_value, load_unit)
        if load_txt:
            self._draw_load_txt(pos_x, pos_y, sub_x, sub_y, load_txt)
        self._draw_load_name(pos_x, pos_y, str(load_id))
        load_dir_x, load_dir_y = pltu.norm_from_points(sub_x, sub_y,
                                                       pos_x, pos_y)
        self._draw_load_bus(sub_x, sub_y, load_dir_x, load_dir_y, load_bus)

    def update_load(self, figure, observation,
                    load_id, load_name, load_bus,
                    load_value, load_unit,
                    pos_x, pos_y,
                    sub_x, sub_y):
        pass

    def _draw_gen_txt(self, pos_x, pos_y, sub_x, sub_y, text):
        dir_x, dir_y = pltu.vec_from_points(sub_x, sub_y, pos_x, pos_y)
        off_x, off_y = pltu.norm_from_points(sub_x, sub_y, pos_x, pos_y)
        txt_x = pos_x + off_x * self._gen_radius
        txt_y = pos_y + off_y * self._gen_radius
        ha = self._h_textpos_from_dir(dir_x, dir_y)
        va = self._v_textpos_from_dir(dir_x, dir_y)
        self.ax.text(txt_x, txt_y, text,
                     color=self._gen_txt_color,
                     wrap=True,
                     fontsize='small',
                     horizontalalignment=ha,
                     verticalalignment=va)

    def _draw_gen_circle(self, pos_x, pos_y):
        patch = patches.Circle((pos_x, pos_y),
                               radius=self._gen_radius,
                               edgecolor=self._gen_edge_color,
                               facecolor=self._gen_face_color)
        self.ax.add_patch(patch)

    def _draw_gen_line(self, pos_x, pos_y, sub_x, sub_y):
        codes = [
            Path.MOVETO,
            Path.LINETO
        ]
        verts = [
            (pos_x, pos_y),
            (sub_x, sub_y)
        ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path,
                                  color=self._gen_line_color,
                                  lw=self._load_line_width)
        self.ax.add_patch(patch)

    def _draw_gen_name(self, pos_x, pos_y, txt):
        self.ax.text(pos_x, pos_y, txt,
                     color=self._gen_txt_color,
                     va='center', ha='center',
                     fontsize='x-small')

    def _draw_gen_bus(self,
                      pos_x, pos_y,
                      norm_dir_x, norm_dir_y,
                      bus_id):
        center_x = pos_x + norm_dir_x * self._sub_radius
        center_y = pos_y + norm_dir_y * self._sub_radius
        face_color = self._line_bus_face_colors[bus_id]
        patch = patches.Circle((center_x, center_y),
                               radius=self._line_bus_radius,
                               facecolor=face_color)
        self.ax.add_patch(patch)

    def draw_gen(self, figure, observation,
                 gen_id, gen_name, gen_bus,
                 gen_value, gen_unit,
                 pos_x, pos_y,
                 sub_x, sub_y):

        self.gen_coordinates[gen_id] = (pos_x, pos_y)

        self.xlim[0] = min(self.xlim[0], pos_x - self._gen_radius)
        self.xlim[1] = max(self.xlim[1], pos_x + self._gen_radius)
        self.ylim[0] = min(self.ylim[0], pos_y - self._gen_radius)
        self.ylim[1] = max(self.ylim[1], pos_y + self._gen_radius)
        self._draw_gen_line(pos_x, pos_y, sub_x, sub_y)
        self._draw_gen_circle(pos_x, pos_y)
        gen_txt = ""
        if self._gen_name:
            gen_txt += "\"{}\":\n".format(gen_name)
        if self._gen_id:
            gen_txt += "id: {}\n".format(gen_id)
        if gen_value is not None:
            gen_txt += pltu.format_value_unit(gen_value, gen_unit)
        if gen_txt:
            self._draw_gen_txt(pos_x, pos_y, sub_x, sub_y, gen_txt)
        self._draw_gen_name(pos_x, pos_y, str(gen_id))
        gen_dir_x, gen_dir_y = pltu.norm_from_points(sub_x, sub_y,
                                                     pos_x, pos_y)
        self._draw_gen_bus(sub_x, sub_y, gen_dir_x, gen_dir_y, gen_bus)

    def update_gen(self, figure, observation,
                   gen_id, gen_name, gen_bus,
                   gen_value, gen_unit,
                   pos_x, pos_y,
                   sub_x, sub_y):
        pass

    def _draw_powerline_txt(self,
                            pos_or_x, pos_or_y,
                            pos_ex_x, pos_ex_y,
                            text):
        pos_x, pos_y = pltu.middle_from_points(pos_or_x, pos_or_y,
                                               pos_ex_x, pos_ex_y)
        off_x, off_y = pltu.orth_norm_from_points(pos_or_x, pos_or_y,
                                                  pos_ex_x, pos_ex_y)
        txt_x = pos_x + off_x * (self._load_radius / 2)
        txt_y = pos_y + off_y * (self._load_radius / 2)
        ha = self._h_textpos_from_dir(off_x, off_y)
        va = self._v_textpos_from_dir(off_x, off_y)
        self.ax.text(txt_x, txt_y, text,
                     color=self._gen_txt_color,
                     fontsize='small',
                     horizontalalignment=ha,
                     verticalalignment=va)

    def _draw_powerline_line(self,
                             pos_or_x, pos_or_y,
                             pos_ex_x, pos_ex_y,
                             color, line_style):
        codes = [
            Path.MOVETO,
            Path.LINETO
        ]
        verts = [
            (pos_or_x, pos_or_y),
            (pos_ex_x, pos_ex_y)
        ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path,
                                  color=color,
                                  lw=self._line_color_width,
                                  ls=line_style)
        self.ax.add_patch(patch)

    def _draw_powerline_bus(self,
                            pos_x, pos_y,
                            norm_dir_x, norm_dir_y,
                            bus_id):
        center_x = pos_x + norm_dir_x * self._sub_radius
        center_y = pos_y + norm_dir_y * self._sub_radius
        face_color = self._line_bus_face_colors[bus_id]
        patch = patches.Circle((center_x, center_y),
                               radius=self._line_bus_radius,
                               facecolor=face_color)
        self.ax.add_patch(patch)

    def _draw_powerline_arrow(self,
                              pos_or_x, pos_or_y,
                              pos_ex_x, pos_ex_y,
                              color, watt_value):
        sign = 1.0 if watt_value > 0.0 else -1.0
        off = 1.0 if watt_value > 0.0 else 2.0
        dx, dy = pltu.norm_from_points(pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)
        lx = dx * self._line_arrow_len
        ly = dy * self._line_arrow_len
        arr_x = pos_or_x + dx * self._sub_radius + off * lx
        arr_y = pos_or_y + dy * self._sub_radius + off * ly
        patch = patches.FancyArrow(arr_x, arr_y,
                                   sign * lx,
                                   sign * ly,
                                   length_includes_head=True,
                                   head_length=self._line_arrow_len,
                                   head_width=self._line_arrow_width,
                                   edgecolor=color,
                                   facecolor=color)
        self.ax.add_patch(patch)

    def draw_powerline(self, figure, observation,
                       line_id, line_name, connected,
                       line_value, line_unit,
                       or_bus, pos_or_x, pos_or_y,
                       ex_bus, pos_ex_x, pos_ex_y):
        # record positions of lines
        self.line_coordinates[line_id] = (pos_or_x, pos_or_y, pos_ex_x, pos_ex_y)

        rho = observation.rho[line_id]
        n_colors = len(self._line_color_scheme) - 1
        color_idx = max(0, min(n_colors, int(rho * n_colors)))
        color = "black"
        if connected and rho > 0.0:
            color = self._line_color_scheme[color_idx]
        line_style = "-" if connected else "--"
        self._draw_powerline_line(pos_or_x, pos_or_y,
                                  pos_ex_x, pos_ex_y,
                                  color, line_style)
        # Deal with line text configurations
        txt = ""
        if self._line_name:
            txt += "\"{}\"\n".format(line_name)
        if self._line_id:
            txt += "id: {}\n".format(str(line_id))
        if line_value is not None:
            txt += pltu.format_value_unit(line_value, line_unit)
        if txt:
            self._draw_powerline_txt(pos_or_x, pos_or_y,
                                     pos_ex_x, pos_ex_y,
                                     txt)

        or_dir_x, or_dir_y = pltu.norm_from_points(pos_or_x, pos_or_y,
                                                   pos_ex_x, pos_ex_y)
        self._draw_powerline_bus(pos_or_x, pos_or_y,
                                 or_dir_x, or_dir_y,
                                 or_bus)
        ex_dir_x, ex_dir_y = pltu.norm_from_points(pos_ex_x, pos_ex_y,
                                                   pos_or_x, pos_or_y)
        self._draw_powerline_bus(pos_ex_x, pos_ex_y,
                                 ex_dir_x, ex_dir_y,
                                 ex_bus)
        watt_value = observation.p_or[line_id]
        if rho > 0.0 and watt_value != 0.0:
            self._draw_powerline_arrow(pos_or_x, pos_or_y,
                                       pos_ex_x, pos_ex_y,
                                       color, watt_value)

    def update_powerline(self, figure, observation,
                         line_id, line_name, connected,
                         line_value, line_unit,
                         or_bus, pos_or_x, pos_or_y,
                         ex_bus, pos_ex_x, pos_ex_y):
        pass

    def draw_legend(self, figure, observation):
        title_str = observation.env_name
        if hasattr(observation, 'month'):
            title_str = "{:02d}/{:02d} {:02d}:{:02d}".format(
                observation.day,
                observation.month,
                observation.hour_of_day,
                observation.minute_of_hour)
        legend_help = [
            Line2D([0], [0], color="black", lw=1),
            Line2D([0], [0], color=self._sub_edge_color, lw=3),
            Line2D([0], [0], color=self._load_edge_color, lw=3),
            Line2D([0], [0], color=self._gen_edge_color, lw=3),
            Line2D([0], [0], marker='o', color=self._line_bus_face_colors[0]),
            Line2D([0], [0], marker='o', color=self._line_bus_face_colors[1]),
            Line2D([0], [0], marker='o', color=self._line_bus_face_colors[2])
        ]
        self.ax.legend(legend_help, [
            "powerline",
            "substation",
            "load",
            "generator",
            "no bus",
            "bus 1",
            "bus 2"
        ], title=title_str)
        # Hide axis
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        # Hide frame
        self.ax.set(frame_on=False)

    def plot_postprocess(self, figure, observation, update):
        if not update:
            xmin = self.xlim[0] - self.xpad
            xmax = self.xlim[1] + self.xpad
            self.ax.set_xlim(xmin, xmax)
            ymin = self.ylim[0] - self.ypad
            ymax = self.ylim[1] + self.ypad
            self.ax.set_ylim(ymin, ymax)
            # self.ax.autoscale(enable=False, tight=True)
            # self.ax.autoscale_view(scalex=False, scaley=False, tight=True)
            # figure.tight_layout()
