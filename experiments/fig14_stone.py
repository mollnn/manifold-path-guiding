import os
test_id = "fig14_stone"
from common import *

scene_dir = "../scenes/stone/"
scene_filename = "stone_reproduce.xml"
gt_filename = "refs/stone.exr"

timeout_list = [180, 300]
spp = 999999

output_name_list = []

for timeout in timeout_list:
    # group = "SMSstar"
    # output_name = f"{test_id}_{group}_{timeout}"
    # output_name_list.append(output_name)
    # cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    # cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    # cmd += f"-Dtrain_auto=false "
    # run_cmd(test_id, cmd, output_name)

    group = "MPG-KNN"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=true "
    cmd += f"-Dspatial_struct=0 "
    run_cmd(test_id, cmd, output_name)

    group = "MPG-M19"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=true "
    cmd += f"-Dspatial_struct=2 "
    cmd += f"-Ddirectional_struct=1 "
    cmd += f"-Dspatial_filter=-1 "
    cmd += f"-Dknn_k=-4000 "
    run_cmd(test_id, cmd, output_name)

    group = "MPG-Ours"
    output_name = f"{test_id}_{group}_{timeout}"
    output_name_list.append(output_name)
    cmd = mitsuba2_cmd + scene_dir + scene_filename + f" -o results/{test_id}/{output_name}.exr "
    cmd += f"-Dspp={spp} -Dtimeout={timeout} "
    cmd += f"-Dtrain_auto=true "
    run_cmd(test_id, cmd, output_name)

import common_imgproc 
common_imgproc.draw(test_id, timeout_list, output_name_list, gt_filename, vmax=1e2)