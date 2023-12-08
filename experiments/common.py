import os
import time
from subprocess import PIPE, run

def run_cmd(testid, command, name):
    print(f"Running {testid} {name}")
    try:
        os.mkdir('results')
    except:
        pass
    try:
        os.mkdir('results' + '/' + testid)
    except:
        pass
    t0 = time.time()
    result = run(command, stdout=PIPE, stderr=PIPE,
                 universal_newlines=True, shell=True)
    log_str = result.stdout
    err_str = result.stderr
    t1 = time.time()
    with open('results/{}_log.txt'.format(testid + "/" + name), 'w') as file:
        file.write(log_str)
        print("@timeused", t1-t0, file=file)
        file.close()
    with open('results/{}_err.txt'.format(testid + "/" + name), 'w') as file:
        file.write(err_str)
        file.close()

mitsuba2_cmd = "..\\mitsuba\\build\\dist\\mitsuba.exe -m scalar_rgb "
