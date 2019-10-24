import os
import sys
import numpy as np
import argparse
import time
import subprocess

def check_start(r=None):
    if r is None:
        r = int(input('Please enter run number where to start: '))
    return r

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Where to start?')
    parser.add_argument('-r','--start', help='Where to start?',type=int,default=None)
    parser.add_argument('-n','--procs', help='Number of processes for parallel lit pixel finding',type=int,default=50)
    parser.add_argument('-m','--module', help='Module number for litpixels',type=int,default=4)
    parser.add_argument('-t','--threshold', help='ADU threshold',type=int,default=25)

    args = parser.parse_args()
    run_num = args.start
    nprocs = args.procs
    module = args.module
    threshold = args.threshold
    
    #p = subprocess.Popen("module load anaconda/3",shell=True)

    if run_num is None:
        run_num = check_start(run_num)
    
    while(1):
        check_dir = os.path.exists('/gpfs/exfel/exp/SPB/201901/p002316/raw/r%.4d/'%run_num)
        if check_dir:
            print('Directory exists')
        else:
            while not check_dir:
                check_dir = os.path.exists('/gpfs/exfel/exp/SPB/201901/p002316/raw/r%.4d/'%run_num)
                time.sleep(60)        
        
        vds_path = '/gpfs/exfel/exp/SPB/201901/p002316/vds/'
        check_vds = os.path.exists(vds_path+'r%.4d_vds_raw.h5'%run_num)
        if check_vds:
            print('vds file already exists!')
        else:
            time_counter = 0
            dir_size = -1
            while time_counter < 5:
                tmp = sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))
                if tmp == dir_size:
                    time_counter += 1 
                    time.sleep(60)
                else:
                    dir_size = tmp
                    time.sleep(60)
            print('Run vds.py for run {}'.format(run_num))
            os.system('python vds.py -o s% %'%(vds_path,run_num))
            while not check_vds:
                check_vds = os.path.exists(vds_path)
                time.sleep(60)
                        
        litpixel_path = '/gpfs/exfel/exp/SPB/201901/p002316/hitlist/'
        check_litpixel = os.path.exists(litpixel_path+'r%.4d_hits.h5'%run_num)
        if check_litpixel:
            print('r%.4d_hits.h5 already exists!'%run_num)
        else:
            print('Run litpixels.py for run {}'.format(run_num))
            os.system('python litpixels.py -n {} -m {} -t {} -o {} {}'.format(nprocs,module,threshold,litpixel_path,vds_path+'r%.4d_vds_raw.h5'%run_num))
        
        print('Done for run {}'.format(run_num))
        run_num = run_num + 1

        
