
import numpy as np
import scipy.spatial


def _min_max_edge(lon_tr_bounds, lat_tr_bounds):
    # Get the largest and smallest edge length in the grid
    lat_diff = np.diff(lat_tr_bounds)
    lon_diff = np.diff(lon_tr_bounds)
    dist = np.sqrt(lat_diff**2 + lon_diff**2)
    dist = np.ma.masked_equal(dist, 0.0)
    return dist.min(), dist.max()


def _nearest_point(kdtree, point, shape, segs):
    # Find the nearest point to the given point.
    # Handles co-located points.

    # Get the nearest 3 points.
    near_dists, near_indices = kdtree.query(point, k=3)#, distance_upper_bound=max_edge)

    # Isolate those that are the same.
    near_dists = np.array(near_dists)
    use_these = np.where(near_dists == near_dists[0])
    near_dists = near_dists[use_these]
    near_indices = near_indices[use_these]
    
    # More than one nearest point?
    if len(near_dists) > 1:
        # Get the ij index for each.
        near_ijs = []
        for near_i in near_indices: 
            near_ij = np.array((int(near_i / shape[1]), near_i % shape[1]))
            near_ijs.append(near_ij)

        # Select nearest ij to last point on the current line.
        if len(segs[-1]) > 0:
            diffs = np.sum(np.abs(near_ijs - segs[-1][-1]), axis=1)
            min_i = np.where(diffs == diffs.min())[0]
            near_ij = near_ijs[min_i]
            near_dist = near_dists[min_i]
        else:
            warnings.warn("can't check last point on a new path")
            near_dist = near_dists[0]
            near_ij = near_ijs[0]
        
    # Just one nearest point.
    else:
        near_i = near_indices[0]
        near_dist = near_dists[0]
        near_ij = np.array((int(near_i / shape[1]), near_i % shape[1]))

#     TODO: I think we need to do this...
#     near_ij[0] -= 1
#     near_ij[1] -= 1

    return near_ij, near_dist


def add_point(segs, new_ij):
    if len(segs[-1]) > 0 and np.all(new_ij == segs[-1][-1]):
        raise ValueError("not adding duplicate" + str(new_ij))
    segs[-1].append(new_ij)
    sanity_check_last_points(segs)
    segs[-1] = remove_spike(segs[-1])
    sanity_check_last_points(segs)


def go_near(segs, target_ij):
    # Add points until we're within 1 cell from target.
    # Assumes one dimension is already within 1 cell away.
    # Expect to end up on an opposite corner to the target.
    ijdiff = target_ij - segs[-1][-1]
    assert(np.any(np.abs(ijdiff) <=1))
    
    # If we're not already near...
    if np.any(np.abs(ijdiff) > 1):
        walk_back_dim = np.where(np.abs(ijdiff) > 1)[0]
        dir = 1 if ijdiff[walk_back_dim] > 0 else -1
        num_walkback_steps = abs(ijdiff[walk_back_dim]) - 1
        walkback_ij_start = segs[-1][-1].copy()
        for walkback_step in range(num_walkback_steps):
            walkback_ij = walkback_ij_start.copy()
            walkback_ij[walk_back_dim] += dir * (walkback_step+1)
            add_point(segs, walkback_ij)
        
    # Check we're within 1 away
    ijdiff = target_ij - segs[-1][-1]
    assert(np.all(np.abs(ijdiff) <= 1))
   

def _nearest_corner(point, prev_ij, curr_ij, lon_tr_bounds, lat_tr_bounds):
    ij1 = (prev_ij[0], curr_ij[1])
    ij2 = (curr_ij[0], prev_ij[1])

    pnt1 = np.array((lon_tr_bounds[ij1], lat_tr_bounds[ij1]))
    pnt2 = np.array((lon_tr_bounds[ij2], lat_tr_bounds[ij2]))

    dist1 = np.sqrt(np.sum((pnt1 - point)**2))
    dist2 = np.sqrt(np.sum((pnt2 - point)**2))

    return np.array(ij1 if dist1 < dist2 else ij2)


def check_add_opposite(segs, target_ij, via_near_point, x_bounds, y_bounds):
    # Are we on the opposite corner to the target?
    ijdiff = target_ij - segs[-1][-1]
    if np.all(np.abs(ijdiff) == 1):
        # Add nearest corner so we're one edge away.
        new_ij = _nearest_corner(via_near_point, segs[-1][-1], target_ij,
                                 x_bounds, y_bounds)
        add_point(segs, new_ij)


def remove_spike(path):
    # Remove the end of the path if it goes back on itself.
    if len(path) >= 3:
        if np.array_equal(path[-1], path[-3]):
            path = path[:-2]
    return path


def sanity_check_last_points(segs):
    # Check the end of the path is a single ij step.
    if len(segs[-1]) >= 2:
        ij_diff = segs[-1][-2] - segs[-1][-1]
        if np.sum(np.abs(ij_diff)) != 1:
            seg = segs[-1]
            raise Exception(str(seg[-(min(10,len(seg))):]))


# TODO: Make sure we can handle >2D cubes (currently fails, I think)
def find_path(cube, line, debug_ax=None):
    start, end = line

    # top right corners
    y_coord = cube.coord(axis="Y")
    x_coord = cube.coord(axis="x")
    lat_tr_bounds = y_coord.bounds[..., 2].squeeze()  # TODO: no squeeze
    lon_tr_bounds = x_coord.bounds[..., 2].squeeze()
    shape = lat_tr_bounds.shape

    # put the corners into a kdtree for fast nearest point searching
    ll_pairs = zip(lon_tr_bounds.flatten(), lat_tr_bounds.flatten())
    kdtree = scipy.spatial.cKDTree(ll_pairs)

    # walk the line
    min_edge, max_edge = _min_max_edge(lon_tr_bounds, lat_tr_bounds)

    # TODO: adaptive step size?
    # for lataitude = 80
    # 0.1:    pointwalk 80 [160, 9, 1, 1, 357, 3, 210]
    # 0.01:   pointwalk 80 [156, 9, 1, 1, 355, 3, 200]
    # 0.001:  pointwalk 80 [156, 9, 1, 1, 355, 3, 200]
    # 0.0001: pointwalk 80 [156, 9, 1, 1, 355, 3, 200]
    step = max(min_edge / 10.0, 0.1)

    dist_vect = end - start
    line_len = np.sqrt(np.sum(dist_vect**2))
    num_steps = int(line_len / step)  # TODO: do the full length of the line
    
    # List of path tuples
    segs = [[]]

    # Find closest corner points to line
    d = dist_vect / num_steps
    if debug_ax:
        debug_start = 2550
        debug_end = debug_start + 50
        traversed_x = [None] * (debug_end - debug_start + 1)
        traversed_y = [None] * (debug_end - debug_start + 1)
        visited = []
        
    # walk
    for i in range(num_steps):
        walk_point = start + d*i
        near_ij, near_dist = _nearest_point(kdtree, walk_point, shape, segs)
        
        if np.all(near_ij == [977, 406]):
#         if i - debug_start == 16:
            pass
        
        if debug_ax and debug_start <= i <= debug_end:
            traversed_x[i-debug_start] = walk_point[0]
            traversed_y[i-debug_start] = walk_point[1]

        # Outside the grid?
        if not np.isfinite(near_dist):
            continue 

        # Before we add the new point, make sure we're next to it...
        if len(segs[-1]) > 0:

            # Are we still near the same point as last time?
            ijdiff = near_ij - segs[-1][-1]
            if np.all(ijdiff == 0):
                continue
            
            # Far enough away to start a new segment?
            if np.all(np.abs(ijdiff) > 2):
                segs.append([])
            
            else:
                
                # More than 1 cell away on BOTH i and j?
                if np.all(np.abs(ijdiff) > 1):
                    # This should get us within one cell of target i OR j.
                    mid_ij = segs[-1][-1] + ijdiff/2
                    go_near(segs, mid_ij)
                    check_add_opposite(segs, mid_ij, walk_point,
                                       lon_tr_bounds, lat_tr_bounds)
                    add_point(segs, mid_ij)
    
                # More than one cell away on EITHER i or j?
                ijdiff = near_ij - segs[-1][-1]
                if np.any(np.abs(ijdiff) > 1):
                    # Make sure we're close on t'other dimension.
                    if not np.abs(ijdiff).min() <= 1:
                        raise Exception("um...")
                    # This should get us within one cell of target i AND j.
                    go_near(segs, near_ij)
                    
                # Excatly one cell away on both i and j?
                check_add_opposite(segs, near_ij, walk_point,
                                   lon_tr_bounds, lat_tr_bounds)
        
        # Now we can add the new point.
        # We should be certain we're one step away now.
        add_point(segs, near_ij)

        if debug_ax and debug_start <= i <= debug_end:
            visited.append((i - debug_start, near_ij))
            seg = segs[-1]
            lat = lat_tr_bounds[near_ij[0], near_ij[1]] 
            lon = lon_tr_bounds[near_ij[0], near_ij[1]] 
            print "{} ({:.3f},{:.3f}) : {}".format(i - debug_start, lon, lat,
                                                   seg[-min(10,len(seg)):])

    # sanity check the segs: ensure single u or v steps
    for path in segs:
        for i in range(len(path) - 1):
            ij_diff = path[i] - path[i+1]
            if np.sum(np.abs(ij_diff)) != 1:
                raise Exception("Line walker error " + str(path[i]) + str(path[i+1]))

    if debug_ax:
        debug_ax.plot(traversed_x, traversed_y, "b.")
        for i in range(len(traversed_x)):
            debug_ax.text(traversed_x[i], traversed_y[i], str(i), size="xx-small")
        for i, ij in visited:
            lat = lat_tr_bounds[ij[0], ij[1]] 
            lon = lon_tr_bounds[ij[0], ij[1]] 
            debug_ax.text(lon, lat, "{}: {},{}".format(i, ij[0], ij[1]), size="x-small")

    return segs
