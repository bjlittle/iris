
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import iris
import line_walk
import top_edge

def _up_U(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = start_yx[0]
        x = start_yx[1] - 1
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = end_yx[0]
        x = start_yx[1]
    return y, x

def _down_U(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = end_yx[0]
        x = end_yx[1] - 1
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = end_yx[0]
        x = start_yx[1]
    return y, x

def _right_V(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = start_yx[0] - 1
        x = start_yx[1]
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = start_yx[0]
        x = end_yx[1]
    return y, x

def _left_V(start_yx, end_yx, grid_type):
    if grid_type == 'C':
        y = end_yx[0] - 1
        x = end_yx[1]
    elif grid_type == 'B1':
        y = start_yx[0]
        x = start_yx[1]
    elif grid_type == 'B2':
        y = start_yx[0]
        x = end_yx[1]
    return y, x

def _get_points(path, input_data, grid_type):
    u, v = input_data["u"], input_data["v"]
    dx, dy = input_data["dx"], input_data["dy"]
    dzu, dzv = input_data["dzu"], input_data["dzv"]

    # Determine the total number of edges in the path.
    n = sum(len(sub_path) - 1 for sub_path in path)

    # Prepare empty arrays for our results
    uv = ma.empty((u.shape[:-2] + (n,)))
    dxdy = ma.empty(n)
    dz = ma.empty((u.shape[-3], n))

    ni = 0
    for sub_path in path:
        for start_yx, end_yx in zip(sub_path[:-1], sub_path[1:]):
            if not (0 <= start_yx[0] <= u.shape[-2] and
                    0 <= start_yx[1] <= u.shape[-1]):
                msg = 'Invalid sub-path point: {}'.format(start_yx)
                raise ValueError(msg)

            # Up => U
            if start_yx[0] + 1 == end_yx[0] and start_yx[1] == end_yx[1]:
                y, x = _up_U(start_yx, end_yx, grid_type)
                scale = 1
                uv_src = u
                dxdy_src = dy
                dz_src = dzu
            # Down => -U
            elif start_yx[0] - 1 == end_yx[0] and start_yx[1] == end_yx[1]:
                y, x = _down_U(start_yx, end_yx, grid_type)
                scale = -1
                uv_src = u
                dxdy_src = dy
                dz_src = dzu
            # Right => -V
            elif start_yx[1] + 1 == end_yx[1] and start_yx[0] == end_yx[0]:
                y, x = _right_V(start_yx, end_yx, grid_type)
                scale = -1
                uv_src = v
                dxdy_src = dx
                dz_src = dzv
            # Left => V
            elif start_yx[1] - 1 == end_yx[1] and start_yx[0] == end_yx[0]:
                y, x = _left_V(start_yx, end_yx, grid_type)
                scale = 1
                uv_src = v
                dxdy_src = dx
                dz_src = dzv
            else:
                msg = 'Invalid sub-path segment: ' \
                    '{0} -> {1})'.format(start_yx, end_yx)
                raise RuntimeError(msg)

            uv[..., ni] = scale * uv_src[..., y, x]
            dxdy[ni] = dxdy_src[y, x]
            dz[:, ni] = dz_src[:, y, x]
            ni += 1

    return {"uv":uv, "dxdy":dxdy, "dz":dz}


def get_path_data(input_data, path, grid_type='C'):

    grid_type = grid_type.upper()
    if grid_type not in ['B', 'C']:
        raise ValueError('Invalid grid type {!r}.'.format(grid_type))

    region_mask = input_data["region_mask"]
    if region_mask is not None:
        u, v = input_data["u"], input_data["v"]
        u[..., region_mask] = ma.masked
        v[..., region_mask] = ma.masked

    if grid_type == 'C':
        data = _get_points(path, input_data, 'C')
    else:
        # Code doesn't seem to handle returning an array...
        raise Exception("Unhandled grid type?")
        data1 = _get_points(path, input_data, 'B1')
        data2 = _get_points(path, input_data, 'B2')
        data = [data1, data2]

    return data


def _common(input_cubes, path, grid_type='C'):
    
    # We only want data from here on.
    input_data = {k:v.data for k,v in input_cubes.items()}
    
    path_data = get_path_data(input_data, path, grid_type=grid_type)
    uv, dxdy, dz = path_data["uv"], path_data["dxdy"], path_data["dz"] 

    if dz.shape[-1] != uv.shape[-1]:
        dz = dz[:, np.newaxis]
    edge_transport = (uv * dz) + dxdy
    return edge_transport.sum(axis=-1)


def stream_function(input_data, path, grid_type='C'):
    path_transport = _common(input_data, path, grid_type=grid_type)
    return path_transport.cumsum(axis=-1)


def net_transport(input_data, path, grid_type='C'):
    path_transport = _common(input_data, path, grid_type=grid_type)
    return path_transport.sum(axis=-1)











# TODO: Remove this test code
if __name__ == "__main__":
    
    print "Loading"
    
    # Load t, u, v
    tuv_path = '/project/hadgem3/data/anbag/ony/'
    tuv_files = [tuv_path + 'anbago_1y_19781201_19791130_grid_T.nc',
                 tuv_path + 'anbago_1y_19781201_19791130_grid_U.nc',
                 tuv_path + 'anbago_1y_19781201_19791130_grid_V.nc']
    tuv_phenom = ['sea_water_potential_temperature',
                  'sea_water_x_velocity',
                  'sea_water_y_velocity']
    t, u, v = iris.load_cubes(tuv_files, tuv_phenom)
   
    # Load region mask
    region_mask_file = '/project/ujcc/CDFTOOLS/mesh_ORCA025L75/' \
    'subbasins_orca025_070909_rename.nc'
    region_mask_phenom = 'tmaskatl'
    region_mask = iris.load_cube(region_mask_file, region_mask_phenom)
    region_mask.data = ~region_mask.data.astype(np.bool)
    
    # Load mesh cubes
    mesh_file = '/project/ujcc/CDFTOOLS/mesh_ORCA025L75/mesh_mask_GO5.nc'
    mesh_phenom = ['e1v', 'e2v', 'e3u', 'e3v']
    dx, dy, dzu, dzv = iris.load_cubes(mesh_file, mesh_phenom)
    
    # cut it down before loading
    t = t[:, :2]
    u = u[:, :2]
    v = v[:, :2]
    dzu = dzu[:2]
    dzv = dzv[:2]
    

    # Lump it all together.
    input_cubes = {"t": t, "u": u, "v": v, "region_mask": region_mask,
                   "dx": dx, "dy": dy, "dzu": dzu, "dzv": dzv}




    checkers = np.zeros(t.shape[-2:], dtype=int)
    checkers[::2, ::2] = 1
    checkers[1::2, 1::2] = 1
    checkers[0,0] = 8


    do_ij_plot = False
    
    
    def plot_ijpath(path, col):
        for seg in path:
            seg = np.array(seg)
            ij_ax.plot(seg[:,1], seg[:,0], c=col)

#     def plot_llpath(paths):
#         for seg in paths:
#             lats = np.ndarray((len(seg), 2))
#             lons = np.ndarray((len(seg), 2))
#             for i, ij in enumerate(seg):
#                 lons[i] = lon_tr[ij[0], ij[1]]
#                 lats[i] = lat_tr[ij[0], ij[1]]
#             ll_ax.plot(lons, lats, c="green")

    def plot_llpath(paths, col):
        # Doesn't draw lines that wrap, so not drawing the whole path in ll.
        for seg in paths:
            lats = []
            lons = []
            for i, ij in enumerate(seg):
                # TODO: Investigate why we need this (bad indexing otherwise)...
                lon = lon_tr[ij[0]-1, ij[1]-1]
                lat = lat_tr[ij[0]-1, ij[1]-1]
                if len(lats) == 0 or (abs(lon - lons[-1]) < 90):
                    lons.append(lon)
                    lats.append(lat)
                else:
                    if len(lats) != 0:
                        ll_ax.plot(lons, lats, c=col)
                    lats = []
                    lons = []
            ll_ax.plot(lons, lats, c=col)


    if do_ij_plot:
        ij_ax = plt.subplot(211)
        ij_ax.pcolormesh(checkers, cmap="binary")
        ll_ax = plt.subplot(212, aspect="equal")
    else:
        ll_ax = plt.axes(aspect="equal")
        
    lon_coord = t.coord('longitude')
    lat_coord = t.coord('latitude')
    ll_ax.pcolormesh(lon_coord.bounds[..., 2], lat_coord.bounds[..., 2], checkers, cmap="binary")
    lon_tr = lon_coord.bounds[..., 2]
    lat_tr = lat_coord.bounds[..., 2]
    
    ### Run the calculations ###
    
    lats = [85]#[65, 75, 85]
    for lat in lats:
        input_line = [np.array((-180.0, lat)), np.array((180.0, lat))]
        ll_ax.plot([-180, 180], [lat, lat], c="blue")

        path = line_walk.find_path(t, line=input_line, debug_ax=ll_ax)
        print [len(seg) for seg in path]
        if do_ij_plot:
            plot_ijpath(path, "green")
        plot_llpath(path, "green")
# 
#         path = top_edge.find_path(t, lat=lat)
#         if do_ij_plot:
#             plot_ijpath(path, "red")
#         plot_llpath(path, "red")


    
#         print "stream_function"
#         sf = stream_function(input_cubes, path)
#     
#         print "net_transport"
#         nt = net_transport(input_cubes, path)

    plt.show()
    print "finished"

