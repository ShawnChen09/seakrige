import contextily as cx
import geopandas as gpd
from libpysal import cg, weights
import math
import networkx as nx
import numpy as np
from scipy.spatial import distance, KDTree
from shapely.geometry import Polygon, LineString, Point


class SeaPath:
    def __init__(self, shapefile, resolution = 0.05, grid_shape = "square", w_method = "queen", k = 8):
        """
        :param shapefile: path to the shapefile.
        :param resolution: resolution of the grid.
        :param grid_shape: shape of the grid, either "triangle" or "square".
        :param w_method: method for creating the weights, "queen", "rook", or "knn".
        :param k: number of nearest neighbors for the weights. Only used if w_method is "knn".
        """
        self._load_gdf(shapefile)
        self._get_coords_from_gdf(r = resolution, shape = grid_shape)
        self._create_G_from_coords(w=w_method, k=k)
        self._filter_G()
        self.kdtree = KDTree(list(self.pos.values()))

    def _load_gdf(self, shapefile):
        self.gdf = gpd.read_file(shapefile)
        self.crs = self.gdf.crs
    
    def _get_coords_from_gdf(self, r, shape):
        xmin, ymin, xmax, ymax = self.gdf.total_bounds
        spacing = min((xmax - xmin) * r, (ymax - ymin) * r)

        if shape == "triangle":
            x_extend, y_extend = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1

            dx = spacing  # Horizontal distance between points
            dy = dx * math.sqrt(3) / 2  # Vertical distance between rows (based on equilateral triangle geometry)

            xcoords = np.arange(xmin - x_extend, xmax + x_extend, dx)
            ycoords = np.arange(ymin - y_extend, ymax + y_extend, dy)

            coords = []
            for i, y in enumerate(ycoords):
                if i % 2 == 0:
                    x_offset = 0
                else:
                    x_offset = dx / 2
                coords.extend([(x + x_offset, y) for x in xcoords])

            self.xycoords = np.array(coords)

        elif shape == "square":
            x_extend, y_extend = (xmax-xmin) * 0.1, (ymax-ymin) * 0.1

            xcoords = [i for i in np.arange(xmin - x_extend, xmax + x_extend, spacing)]
            ycoords = [i for i in np.arange(ymin - y_extend, ymax + y_extend, spacing)]

            self.xycoords = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1, 2) # a 2D array like [[x1,y1], [x1,y2], ...

        else:
            raise ValueError("Invalid shape. Choose 'triangle' or 'square'.")

    def _create_G_from_coords(self, w, k):
        self.cells = cg.voronoi_frames(self.xycoords, clip="convex_hull", return_input=False, as_gdf=True)
        if w == "rook":
            self.W = weights.contiguity.Rook.from_dataframe(self.cells, use_index=False)
        elif w == "queen":
            self.W = weights.contiguity.Queen.from_dataframe(self.cells, use_index=False)
        elif w == "knn":
            self.W = weights.distance.KNN.from_dataframe(self.cells, k=k)
        else:
            raise ValueError("Invalid weight method. Choose 'rook', 'queen', or 'knn'.")
        self.G = self.W.to_networkx()
        self.pos = dict(zip(self.G.nodes, self.xycoords))
        self.rev_pos = {tuple(v): k for k, v in self.pos.items()}

    def _filter_G(self):
        p = Polygon(np.array(self.gdf.get_coordinates()))

        for edge in list(self.G.edges):
            line = LineString([self.pos[edge[0]], self.pos[edge[1]]])
            if line.intersects(p):
                self.G.remove_edge(*edge)

        for node in list(self.G.nodes):
            point = Point(np.array(self.pos[node]))
            if point.intersects(p):
                self.G.remove_node(node)

        self.distances = {(u, v): distance.euclidean(self.pos[u], self.pos[v])
                          for u in self.G.nodes for v in self.G.nodes if u != v}

    def _find_node(self, coords):
        _, idx = self.kdtree.query(coords)
        closest_coord = list(self.pos.values())[idx]
        return self.rev_pos[tuple(closest_coord)]

    def _get_shortest_path(self, source, target):
        return nx.dijkstra_path(self.G, source=source, target=target)

    def calc_path_from_G(self, source_coord, target_coord):
        source = self._find_node(source_coord)
        target = self._find_node(target_coord)
        self.path = self._get_shortest_path(source, target)
        self.path_edges = zip(self.path, self.path[1:])

        self.path_length = sum(self.distances[(u, v)] for u, v in self.path_edges)
        return self.path_length

    def plot_geomap(self):
        ax = self.cells.plot(facecolor="lightblue", alpha=0.10,)
        cx.add_basemap(ax, crs=self.crs)

    def plot_G(self, node_size = 1, edge_size = 1):
        nx.draw(self.G,
                self.pos,
                node_size=node_size,
                width=edge_size,
                node_color="k",
                edge_color="k",
                alpha=0.9,)

    def plot_path(self, path_color = 'r', width = 2):
        nx.draw_networkx_edges(self.G,
                               self.pos,
                               edgelist=self.path_edges,
                               edge_color=path_color,
                               width=width,)