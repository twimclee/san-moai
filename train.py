import os

import argparse

from pathfilemgr import MPathFileManager
from hyp_data import MHyp, MData

#######################################################################
# parameter
#######################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--volume', help='volume directory', default='moai')
parser.add_argument('--project', help='project directory', default='test_project')
parser.add_argument('--subproject', help='subproject directory', default='test_subproject')
parser.add_argument('--task', help='task directory', default='test_task')
parser.add_argument('--version', help='version', default='v1')
opt = parser.parse_args()

#######################################################################
# load hyp
#######################################################################
mpfm = MPathFileManager(opt.volume, opt.project, opt.subproject, opt.task, opt.version)
mhyp = MHyp()
mpfm.load_train_hyp(mhyp)
mpfm.save_hyp(mhyp)


#######################################################################
# make dataset
#######################################################################
ocmd = r'python dataset_tool.py --source {} --dest {} --resolution {}'

cmd = ocmd.format(
	mpfm.train_path,
	mpfm.train_data,
	f'{mhyp.imgsize}x{mhyp.imgsize}'
	)
print(cmd)
os.system(cmd)
print("dataset load done")

#######################################################################
# train
#######################################################################
ocmd = r'python run_train.py --outdir {} --data {} --imgsize {} --cfg {} --gpus {} --batch {} --batch_gpu {} --kimg {} --workers {} --snap {} --syn_layers {} --mirror {}'

cmd = ocmd.format(
	mpfm.train_result, 
	mpfm.train_data, 
	mhyp.imgsize, 
	mhyp.cfg, 
	mhyp.gpus, 
	mhyp.batch, 
	mhyp.batch_gpu, 
	mhyp.kimg,
	mhyp.workers,
	mhyp.snap,
	mhyp.syn_layers,
	mhyp.mirror)
print(cmd)
os.system(cmd)
print("train load done")

    # opts.outdir = mpfm.train_result
    # opts.data = f'{mpfm.train_path}/train.zip'
    # opts.imgsize = mhyp.imgsize
    # opts.cfg = mhyp.cfg
    # opts.gpus = mhyp.gpus
    # opts.batch = mhyp.batch
    # opts.batch_gpu = mhyp.batch_gpu
    # opts.kimg = mhyp.kimg
    # opts.workers = mhyp.workers
    # opts.snap = mhyp.snap
    # opts.syn_layers = mhyp.syn_layers
    # opts.mirror = mhyp.mirror