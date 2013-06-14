# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Test the stream trnasport module.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import matplotlib.pyplot as plt
import numpy as np

import iris
import iris.experimental.stream_transport as stream_transport
import iris.experimental.stream_transport.line_walk as line_walk
import iris.experimental.stream_transport.top_edge as top_edge


class TestStreamTransport(tests.IrisTest):
    
    @classmethod
    def setUpClass(cls):

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
        cls.input_cubes = {"t": t, "u": u, "v": v, "region_mask": region_mask,
                           "dx": dx, "dy": dy, "dzu": dzu, "dzv": dzv}

        # For humanns, we visually check paths on a checker-board background.
        cls.checkers = np.zeros(t.shape[-2:], dtype=int)
        cls.checkers[::2, ::2] = 1
        cls.checkers[1::2, 1::2] = 1
        cls.checkers[0,0] = 8
        
    def integration_test(self, module, path_func_args):
        # We currently have one big integration test for everything
        # because we currently use big data and don't want to
        # recalculate the path in a separate path test.
        
        t = self.input_cubes["t"]
        mod_name = module.__name__.split(".")[-1]

        # Each item in path_func_args defines a latitude/line to test.
        # For each, calculate and test a path through the data.
        plt.pcolormesh(self.checkers, cmap="binary")
        for i, path_func_arg in enumerate(path_func_args):

            # Calculate a path through the data using the given func and arg.
            path = module.find_path(t, path_func_arg)

            # Plot it for humans.
            for seg in path:
                seg = np.array(seg)
                plt.plot(seg[:,1], seg[:,0], c="green", linewidth=2)

            # Check stream/transport calculations along this path.
            sf = stream_transport.stream_function(self.input_cubes, path)
            self.assertString(str(sf), tests.get_result_path((
                "experimental", "stream_transport",
                "{}_stream_{}.txt".format(mod_name, i)))) 
            
            nt = stream_transport.net_transport(self.input_cubes, path)
            self.assertString(str(nt), tests.get_result_path((
                "experimental", "stream_transport",
                "{}_net_{}.txt".format(mod_name, i))))

        self.check_graphic()

    def test_top_edge(self):
        input_lats = [65, 75, 85]
        self.integration_test(top_edge, input_lats)

    def test_line_walk(self):
        input_lats = [65, 75, 85]
        input_lines = [[np.array((-180.0, lat)), np.array((180.0, lat))]
                       for lat in input_lats]
        self.integration_test(line_walk, input_lines)


if __name__ == "__main__":
    tests.main()
