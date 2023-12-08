import os
test_id = "fig08_living"
from common import *

scene_dir = "../scenes/living/"
scene_filename = "living_reproduce.xml"
gt_filename = ""
timeout_list = [120]
print("timeout_list", timeout_list)
print("sum(timeout_list)", sum(timeout_list))
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

    group = "MPG+Selective"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=true "
    cmd += f"-Dselective=true "
    run_cmd(test_id, cmd, output_name)

import common_imgproc 
common_imgproc.draw(test_id, timeout_list, output_name_list, gt_filename, vmax=1e3)