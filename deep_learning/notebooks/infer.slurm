#!/bin/bash
#SBATCH --job-name=ffsl_infer1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --account=rai
#SBATCH --partition=rai-gpu-grn
#SBATCH --qos=rai-gpu-grn
#SBATCH --gres=gpu:h200:1
#SBATCH --time=14-00:00:00
#SBATCH --output=/uufs/chpc.utah.edu/common/home/u0972368/FFSL_HRWUI/deep_learning/logs/infer_%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=u0972368@utah.edu

# =============================================================================
# COUNTY KEY — set COUNTY_NBR below to select which county to run
#  1: Beaver        2: Box_Elder     3: Cache          4: Carbon
#  5: Daggett       6: Davis         7: Duchesne       8: Emery
#  9: Garfield     10: Grand        11: Iron          12: Juab
# 13: Kane         14: Millard      15: Morgan        16: Piute
# 17: Rich         18: Salt_Lake    19: San_Juan      20: Sanpete
# 21: Sevier       22: Summit       23: Tooele        24: Uintah
# 25: Utah         26: Wasatch      27: Washington    28: Wayne
# 29: Weber
#
# Estimated runtimes at 10 it/s (actual speed depends on GPU assigned):
#  Fast  (<50 hrs):  Davis(6)=4.9   Weber(29)=13.9  Salt_Lake(18)=20.2
#                    Cache(3)=35.8  Wayne(28)=34.6  Piute(16)=33.1
#  Medium (50-100h): Carbon(4)  Daggett(5)  Morgan(15)  Rich(17)
#                    Emery(8)   Sanpete(20) Sevier(21)  Wasatch(26)
#  Long  (>100 hrs): Beaver(1)  Box_Elder(2) Garfield(9)  Grand(10)
#                    Iron(11)   Juab(12)    Kane(13)    Millard(14)
#                    Summit(22) Tooele(23)  Uintah(24)  Utah(25)
#                    Washington(27) San_Juan(19)
# =============================================================================
COUNTY_NBR=21

# =============================================================================
# DO NOT EDIT BELOW THIS LINE
# =============================================================================

# Compute county name once — used by both signal handler and end-of-job check
COUNTY_NAME=$(python3 -c "
k={1:'Beaver',2:'Box_Elder',3:'Cache',4:'Carbon',5:'Daggett',6:'Davis',
   7:'Duchesne',8:'Emery',9:'Garfield',10:'Grand',11:'Iron',12:'Juab',
   13:'Kane',14:'Millard',15:'Morgan',16:'Piute',17:'Rich',18:'Salt_Lake',
   19:'San_Juan',20:'Sanpete',21:'Sevier',22:'Summit',23:'Tooele',24:'Uintah',
   25:'Utah',26:'Wasatch',27:'Washington',28:'Wayne',29:'Weber'}
print(k[$COUNTY_NBR])")

CKPT="/uufs/chpc.utah.edu/common/home/u0972368/FFSL_HRWUI/data/predictions/predictions_${COUNTY_NAME}/infer_checkpoint_${COUNTY_NAME}.json"

# Signal handler — fires 5 minutes before wall time hits
# Resubmits this exact script so inference resumes from checkpoint
handle_timeout() {
    echo "Wall time approaching — checking for checkpoint..."
    if [ -f "$CKPT" ]; then
        echo "Checkpoint found — resubmitting for $COUNTY_NAME"
        sbatch $0
    else
        echo "No checkpoint found — job may have already completed"
    fi
}
trap 'handle_timeout' USR1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU found')"
echo "County: $COUNTY_NBR — $COUNTY_NAME"

module load deeplearning/2025.4

echo "Environment ready"
echo "Starting Python: $(date)"

python /uufs/chpc.utah.edu/common/home/u0972368/FFSL_HRWUI/deep_learning/notebooks/infer.py $COUNTY_NBR

# End-of-job check — only reached if Python finished before wall time
if [ -f "$CKPT" ]; then
    echo "Checkpoint found after Python exit — resubmitting for $COUNTY_NAME"
    sbatch $0
else
    echo "No checkpoint — $COUNTY_NAME completed successfully"
fi

echo "Job finished: $(date)"
