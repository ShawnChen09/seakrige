# SeaKrige
The purpose of this project is to implement the Kriging method to interpolate values in the sea with polygon (island) boundaries.
# Getting Started

# Installation
1. Required Python Package Installation
```sh
python -m pip install -r requirements.txt
```
2. Local package installs
```sh
python -m pip install -e .
```

# Usage

## `SeaPath` module
The SeaPath module calculates the Manhattan distance between given source coordinates and target coordinates, ensuring the path only traverses water areas by recognizing the polygon(s) provided from a .shp file.

Example code:

```python
from seakrige import SeaPath

# initialize the SeaPath instance
sea_path = SeaPath(
    shapefile="./tw_shp/tw.shp",  # path to the shapefile
    resolution=0.05,  # resolution of the grid
    grid_shape="square",  # shape of the grid, either "triangle" or "square"
    w_method="queen",  # method for creating the weights, "queen", "rook", or "knn"
    k=8, # number of nearest neighbors for the "knn" weight method. Only used if w_method is "knn"
)

```

```python
pos1 = (120.268875, 24.283269)
pos2 = (121.636672, 23.580348)
# calculate the shortest path (manhattan distance), then convert to geographical distance
sea_path.calc_path_from_G(source_coord = pos1, target_coord = pos2)
# plot geographical map
sea_path.plot_geomap()
# plot grid
sea_path.plot_G(node_size = 1, edge_size = 0.5)
# plot shortest sea path
sea_path.plot_path(path_color = 'r', width = 2)
```

<img src="https://github.com/ShawnChen09/seakrige/raw/main/img/seapath.png" width="300"/>

## `SeaKrige` module
The SeaKrige module performs the interpolation of z-values at unsampled sites based on spatial autocorrelation (using the distance calculated by SeaPath) and visualizes the results.

Example code:

```python
import numpy as np
from seakrige import SeaKrige

longitude = ... # array_like (N)
latitude = ... # array_like (N)
z_values = # array_like (N)
known_points = np.column_stack((longitude, latitude)) # array_like (Nx2)
shapefile = "XXX.shp"

# initialize the instance
sea_krige = SeaKrige(
    known_points,
    z_values,
    shapefile,
    resolution=0.1,
    grid_shape="square",
    w_method="queen",
    k=8,
)
```

```python
sea_krige.plot_scatter(
    # All parameters are optional
    colormap=COLORMAP,
    zoom=ZOOM-RATIO,
    range_color=[Vmin, Vmax],
    center={"lat": LAT, "lon": LON},
    title=TITLE,
    marker_size=SIZE,
    marker_opacity=OPACITY,
    layout_margin= {"r": R, "t": T, "l": L, "b": B},
    layout_width=WIDTH,
    layout_height=HEIGHT,
    html_path=HTMLPATH,
    show_fig=True,
)
```

```python
sea_krige.predict()
```

```python
# plot both semi-variance scatter plot and covariance model fitted line
sea_krige.plot_fit_variogram()
```

```python
# plot interpolation results on a geographical map
sea_krige.plot_predict_result(title=TITLE, vmin=VMIN, vmax=VMAX)
```