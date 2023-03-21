#!/usr/bin/env python3

import os
import re
from tqdm import tqdm
import shutil
import glob

base = '<absolute-path-to-code>/'

archive = os.path.join(base, 'archive')
archive_byexp = os.path.join(base, 'archive_byexp')
logs = os.path.join(base, 'logs')
slurm_logs = os.path.join(base, 'slurmlogs')

def getexperiment(run):
    scriptname = re.match(r'^(.*)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.*$', run).group(1)
    # script = os.path.join(archive, run, 'scripts', '_'.join(run.split('_')[:-2])+'.sh')
    script = os.path.join(archive, run, 'scripts', scriptname+'.sh')
    configpath = None
    with open(script) as f:
        for l in f:
            if '#' in l:
                l = l[:l.index('#')]
            l = l.split()
            if '--config' in l:
                configpath = l[l.index('--config')+1]
                break

    assert configpath is not None, run
    # raise RuntimeError(configpath)
    with open(os.path.join(archive, run, configpath)) as f:
        for l in f:
            if 'expname' in l:
                res = '='.join(l.split('=')[1:]).strip()
                if not os.path.exists(os.path.join(logs, res)):
                    res = None
                return res

print('reading exps...')
runs = list(sorted(os.listdir(archive)))
exps = dict()
for run in runs:
    exp = getexperiment(run)
    if exp is not None:
        exps[exp] = run
    # print(run, exp)

# 1/0
print('updating...')
os.makedirs(archive_byexp, exist_ok=True)

for exp, run in exps.items():
    for link_fn in [os.path.join(archive_byexp, exp), os.path.join(logs, exp, 'archive')]:
        source = os.path.join(archive, run)
        need_to_link = None
        if os.path.exists(link_fn):
            if os.path.realpath(os.readlink(link_fn)) != os.path.realpath(source):
                print('recreating for', os.path.realpath(os.readlink(link_fn)), os.path.realpath(source))
                os.remove(link_fn)
                need_to_link = True
            else:
                need_to_link = False
        else:
                need_to_link = True

        if need_to_link:
            print(link_fn, source)
            try:
                os.symlink(source, link_fn)
            except PermissionError:
                print('permission error')

