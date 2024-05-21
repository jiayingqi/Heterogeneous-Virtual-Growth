"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Ke Liu (liuke@pku.edu.cn)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsor:
- David C. Crawford Faculty Scholar Award from the Department of Civil and
  Environmental Engineering and Grainger College of Engineering at the
  University of Illinois

Citations:
- Jia, Y., Liu, K., Zhang, X.S., 2024. Modulate stress distribution with
  bio-inspired irregular architected materials towards optimal tissue support.
  Nature Communications 15, 4072. https://doi.org/10.1038/s41467-024-47831-2
- Jia, Y., Liu, K., Zhang, X.S., 2024. Topology optimization of irregular
  multiscale structures with tunable responses using a virtual growth rule.
  Computer Methods in Applied Mechanics and Engineering 425, 116864.
  https://doi.org/10.1016/j.cma.2024.116864
"""

import pyvista
import numpy as np


def plot_mesh(elements, cell_types, nodes, color="#96ADFC", show_edges=False,
              line_width=None, render_lines_as_tubes=False,
              plot_nodes=False, point_size=20,
              point_color="maroon", plot_box="2D", box_length=5, view_xy=False,
              figsize=2000, fig_path="", fig_name="Block.jpg"):
    """Plot the mesh."""
    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme("document")
    pyvista.start_xvfb()

    plotter = pyvista.Plotter(window_size=[figsize, figsize])
    grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)
    grid.cell_data["data"] = 0.5
    plotter.add_mesh(grid, color=color, lighting=True,
                     show_edges=show_edges, line_width=line_width,
                     show_scalar_bar=False, render_lines_as_tubes=render_lines_as_tubes)

    if plot_nodes:
        point_cloud = pyvista.PolyData(nodes)
        plotter.add_mesh(point_cloud, color=point_color, point_size=point_size,
                         render_points_as_spheres=True)

    if plot_box in ["2D", "3D"]:
        if plot_box == "2D":
            elements_box = np.array([4, 0, 1, 2, 3])
            cell_types_box = np.array([9])
            nodes_box = np.array([
                [box_length/2, box_length/2, 0],
                [-box_length/2, box_length/2, 0],
                [-box_length/2, -box_length/2, 0],
                [box_length/2, -box_length/2, 0],
            ])
        else:
            elements_box = np.array([8, 0, 1, 2, 3, 4, 5, 6, 7])
            cell_types_box = np.array([12])
            nodes_box = np.array([
                [box_length/2, box_length/2, -box_length/2],
                [-box_length/2, box_length/2, -box_length/2],
                [-box_length/2, -box_length/2, -box_length/2],
                [box_length/2, -box_length/2, -box_length/2],
                [box_length/2, box_length/2, box_length/2],
                [-box_length/2, box_length/2, box_length/2],
                [-box_length/2, -box_length/2, box_length/2],
                [box_length/2, -box_length/2, box_length/2],
            ])
        grid_box = pyvista.UnstructuredGrid(
            elements_box, cell_types_box, nodes_box)
        plotter.add_mesh(grid_box, clim=(0, 1), cmap="Greys", lighting=True,
                         show_edges=False, show_scalar_bar=False,
                         style="wireframe", line_width=6, color="black")

    plotter.background_color = "white"
    if view_xy:
        plotter.view_xy()
    plotter.show_axes()

    plotter.screenshot(fig_path+fig_name, window_size=[figsize, figsize])
    plotter.close()
