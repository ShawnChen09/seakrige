from typing import Literal

import contextily as cx
import geopandas as gpd
import geopandas.tools
import gstools as gs
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from geokrige.tools import TransformerGDF
from scipy.spatial import distance

from seakrige import SeaPath


class SeaKrige(SeaPath):
    def __init__(
        self,
        known_points,
        z,
        shapefile,
        resolution=0.1,
        grid_shape="square",
        w_method="queen",
        k=8,
    ):
        super().__init__(shapefile, resolution, grid_shape, w_method, k)
        self.known_points = known_points
        self.z = z
        self.dist_func = self.calc_path_from_G
        self.models = {
            "Gaussian": gs.Gaussian,
            "Exponential": gs.Exponential,
            "Matern": gs.Matern,
            "Stable": gs.Stable,
            "Rational": gs.Rational,
            "Circular": gs.Circular,
            "Spherical": gs.Spherical,
            "SuperSpherical": gs.SuperSpherical,
            "JBessel": gs.JBessel,
        }
        self._calc_distmx()
        self._calc_semivariance()
        self._bin_data()
        self._fit_variogram()
        self._get_variogram_matrix()

    def plot_scatter(
        self,
        colormap: str = "viridis",
        mapbox_style="open-street-map",
        zoom: int | None = None,
        range_color: tuple[float, float] = None,
        center: dict[Literal["lat", "lon"], float] = {"lat": 24.2, "lon": 120},
        title: str = None,
        marker_size: int = 16,
        marker_opacity: int = 0.7,
        layout_margin: dict[Literal["r", "t", "l", "b"], float] = {
            "r": 0,
            "t": 40,
            "l": 0,
            "b": 0,
        },
        layout_width: float = None,
        layout_height: float = None,
        html_path: str = None,
        show_fig: bool = True,
    ):
        df = {
            "lat": self.known_points[:, 1],
            "lon": self.known_points[:, 0],
            "value": self.z,
        }
        fig = px.scatter_mapbox(
            df,
            lat="lat",
            lon="lon",
            color="value",
            color_continuous_scale=colormap,
            zoom=zoom,
            mapbox_style=mapbox_style,
            title=title,
            range_color=range_color,
            center=center,
        )

        fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

        fig.update_layout(
            margin=layout_margin, width=layout_width, height=layout_height
        )

        if show_fig:
            fig.show()

        if html_path is not None:
            fig.write_html(html_path)

        return fig

    def _calc_distmx(self):
        dist_mx = [
            [
                max(self.dist_func(apoint, bpoint), distance.euclidean(apoint, bpoint))
                for bpoint in self.known_points
            ]
            for apoint in self.known_points
        ]
        self.distmx = np.array(dist_mx)

    def _calc_semivariance(self):
        z = self.z
        z = (z - z.mean()) / (z.std())
        n = len(z)
        self.semivarmx = np.array(
            [[0.5 * (z[i] - z[j]) ** 2 for j in range(n)] for i in range(n)]
        )

    def _bin_data(self, num_bin=20):
        upper_tri_indices = np.triu_indices(len(self.distmx[0]), k=1)
        distances = self.distmx[upper_tri_indices]
        semivariances = self.semivarmx[upper_tri_indices]

        max_distance = distances.max()
        bin_width = max_distance / num_bin
        bins = np.arange(0, max_distance + bin_width, bin_width)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        mean_semivariances = np.zeros(len(bin_centers))
        for i in range(len(bin_centers)):
            in_bin = (distances >= bins[i]) & (distances < bins[i + 1])
            if in_bin.any():
                mean_semivariances[i] = np.mean(semivariances[in_bin])
            else:
                mean_semivariances[i] = np.nan

        valid = ~np.isnan(mean_semivariances)
        self.bin_centers = bin_centers[valid]
        self.gamma = mean_semivariances[valid]

    def _fit_variogram(self):
        fit_models = {}
        scores = {}
        # plt.scatter(bin_centers, mean_semivariances, color="k", label="data")
        # ax = plt.gca()
        for model in self.models:
            fit_model = self.models[model](dim=2)
            _, _, r2 = fit_model.fit_variogram(
                self.bin_centers, self.gamma, return_r2=True
            )
            # fit_model.plot(x_max=max(distances), ax=ax)
            fit_models[model] = fit_model
            scores[model] = r2
        chosen_model = max(scores, key=scores.get)
        self.fit_model = fit_models[chosen_model]

    def plot_fit_variogram(self):
        fig, ax = plt.subplots()
        ax.figure.set_size_inches(8, 6)
        ax.scatter(
            self.distmx.flatten(),
            self.semivarmx.flatten(),
            color="blue",
            label="Experimental Variogram",
        )
        ax.scatter(self.bin_centers, self.gamma, color="red", label="Binned Variogram")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Semivariance")
        ax.set_title("Raw Variogram Scatter Plot")
        ax.legend()
        ax.grid(True)

        ax = self.fit_model.plot(x_max=max(self.distmx.flatten()), color="k")
        ax.scatter(self.bin_centers, self.gamma, color="red", label="Binned Variogram")
        ax.figure.set_size_inches(8, 6)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Semivariance")
        ax.set_title("Fitted Variogram Plot")
        ax.legend()
        ax.grid(True)

    # predict part
    def _get_variogram_matrix(self):
        n = self.distmx.shape[0]
        self.variogram_matrix = np.array(
            [
                [self.fit_model.variogram(self.distmx[i, j]) for j in range(n)]
                for i in range(n)
            ]
        )

    def _predict_kriging(self, unknown_point):
        n = self.known_points.shape[0]
        distances = np.array(
            [self.dist_func(self.known_points[i], unknown_point) for i in range(n)]
        )
        variogram_vector = self.fit_model.variogram(distances)

        try:
            weights = np.linalg.solve(self.variogram_matrix, variogram_vector)
        except np.linalg.LinAlgError:
            weights = np.linalg.pinv(self.variogram_matrix) @ variogram_vector

        predicted_value = np.dot(weights, self.z)
        return predicted_value

    def _transform_gdf_to_grid(self):
        transformer = TransformerGDF()
        transformer.load(self.gdf)

        meshgrid = transformer.meshgrid(density=0.5)
        self.grid_mask = transformer.mask()
        self.X, self.Y = meshgrid

    def predict(self):
        self._transform_gdf_to_grid()
        self.Z = []
        for query_point in zip(self.X.flatten(), self.Y.flatten()):
            try:
                predict_value = self._predict_kriging(
                    query_point,
                )
            except Exception:
                predict_value = 0
            self.Z.append(predict_value)
        self.Z = np.array(self.Z).reshape(self.X.shape)
        self.Z_mask[self.grid_mask] = None

    def plot_predict_result(self, title=None, show_base_map=True, vmin=None, vmax=None):
        fig, ax = plt.subplots()
        self.gdf.plot(
            facecolor="none", edgecolor="black", linewidth=1.5, zorder=5, ax=ax
        )
        cx.add_basemap(ax, crs=self.crs)

        Z_reverse = self.Z_mask[::-1]
        plt.imshow(
            Z_reverse,
            extent=[self.X.min(), self.X.max(), self.Y.min(), self.Y.max()],
            interpolation="gaussian",
            vmin=vmin,
            vmax=vmax,
        )
        cax = fig.add_axes([0.93, 0.134, 0.02, 0.72])
        plt.colorbar(cax=cax, orientation="vertical")
        plt.title(title)
        ax.grid(lw=1)
        ax.set_xlim(min(self.X[0]), max(self.X[0]))
        ax.set_ylim(min(self.Y[0]), max(self.Y[-1]))
