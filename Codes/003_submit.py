import subprocess

for year in range(2020, 2021):
    if year == 2017:
        for month in range(8, 13):
            ym = '%d%02d' % (year, month)
            with open('submit%s.sh' % ym, 'w') as f:
                f.write('''#!/usr/bin/env sh\n#SBATCH --partition=broadwl\n#SBATCH --account=pi-dachxiu\n#SBATCH --job-name=run_%s\n#SBATCH --output=%s.out\n#SBATCH --constraint=ib\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --exclusive\n#SBATCH --time=24:00:00\n\npython 002_run.py %s %s''' % (ym, ym, year, month))
            subprocess.call('sbatch submit%s.sh' % ym, shell=True)
    elif year == 2020:
        for month in range(1, 2):
            ym = '%d%02d' % (year, month)
            with open('submit%s.sh' % ym, 'w') as f:
                f.write('''#!/usr/bin/env sh\n#SBATCH --partition=broadwl\n#SBATCH --account=pi-dachxiu\n#SBATCH --job-name=run_%s\n#SBATCH --output=%s.out\n#SBATCH --constraint=ib\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --exclusive\n#SBATCH --time=24:00:00\n\npython 002_run.py %s %s''' % (ym, ym, year, month))
            subprocess.call('sbatch submit%s.sh' % ym, shell=True)
    else:
        for month in range(1, 13):
            ym = '%d%02d' % (year, month)
            with open('submit%s.sh' % ym, 'w') as f:
                f.write('''#!/usr/bin/env sh\n#SBATCH --partition=broadwl\n#SBATCH --account=pi-dachxiu\n#SBATCH --job-name=run_%s\n#SBATCH --output=%s.out\n#SBATCH --constraint=ib\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=1\n#SBATCH --exclusive\n#SBATCH --time=24:00:00\n\npython 002_run.py %s %s''' % (ym, ym, year, month))
            subprocess.call('sbatch submit%s.sh' % ym, shell=True)
        
#SBATCH --partition=gavoth-sdb
#SBATCH --qos=gavoth


