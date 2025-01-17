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
# train
#######################################################################
ocmd = r'python run_test.py --outdir {} --seeds 0 --batch-sz 1 --network {}'

cmd = ocmd.format(
	mpfm.test_result,
	f'{mpfm.train_result}/network-snapshot.pkl'
	)
print(cmd)
os.system(cmd)
print("test done")

