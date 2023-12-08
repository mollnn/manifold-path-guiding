import os
test_id = "fig07_plane"
from common import *

scene_dir = "../scenes/plane/"
scene_filename = "plane_reproduce_fig07.xml"
gt_filename = "refs/plane.exr"
timeout_list = [60, 120]
spp = 999999

output_name_list = []

for timeout in timeout_list:
    group = "MPG"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=true "
    run_cmd(test_id, cmd, output_name)

    group = "MPG+Prod"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=true "
    cmd += f"-Dproduct_sampling=true "
    run_cmd(test_id, cmd, output_name)

import common_imgproc 
common_imgproc.draw(test_id, timeout_list, output_name_list, gt_filename, vmax=1e3)