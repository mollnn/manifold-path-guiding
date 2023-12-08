import os
test_id = "fig09_lamp"
from common import *

scene_dir = "../scenes/lamp/"
scene_filename = "lamp_reproduce.xml"
gt_filename = "refs/lamp.exr"
timeout_list = [15, 30]
spp = 999999

output_name_list = []

for timeout in timeout_list:
    group = "SMSstar"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=false "
    run_cmd(test_id, cmd, output_name)

    group = "MPG"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=true "
    run_cmd(test_id, cmd, output_name)

import common_imgproc 
common_imgproc.draw(test_id, timeout_list, output_name_list, gt_filename, vmax=1e3)