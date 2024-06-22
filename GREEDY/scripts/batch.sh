#!/bin/bash
#SBATCH -e slurm-%j.err
#SBATCH -o greedy-layout-%j.json
#SBATCH --nodes=1           # Run all processes on a single node
#SBATCH --ntasks=1          # Run a single task
#SBATCH --mem=1G            # Job memory request
#SBATCH --cpus-per-task=1   # Number of CPU cores per task
#SBATCH --gres=gpu:1        # Number of GPUs per task
#SBATCH --time=00:10:00     # Time limit hrs:min:sec

KUERZEL=$1
BARCODE_SET=$2

if [ "$#" -ne 2 ]; then
    echo "incorrect number of arguments ($#), please call this script as follows:"
    echo "batch.sh <KUERZEL> <BARCODE_SET"
    exit 1
fi

# ensure, directory exists
srun mkdir /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}

# copy repository to hp node
sbcast -f ${BARCODE_SET} /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}/`basename ${BARCODE_SET}`
for d in $(find barcode-layout -type d -not -path *git*); do srun mkdir -p /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}/${d}; done
for f in $(find barcode-layout -type f -not -path *git*); do sbcast -f $f /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}/${f}; done

# compile algorithm
srun /usr/local/cuda-12.3/bin/nvcc -Xptxas -O3 --debug --ptxas-options=-v \
    -o /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}/barcode-layout/bin/barlay \
    /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}/barcode-layout/src/main.cu

# execute algorithm
cat /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}/`basename ${BARCODE_SET}` | /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}/barcode-layout/bin/barlay

# cleanup
srun rm -R /zpool1/slurm_data/${KUERZEL}/${SLURM_JOBID}