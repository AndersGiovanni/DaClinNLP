
#!/bin/bash

#SBATCH --job-name=icd_baseline    # Job name
#SBATCH --output=outputs/icd_baseline.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=16       # Schedule one core
#SBATCH --gres=gpu               # Schedule a GPU
#SBATCH --time=05:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=red    # Run on either the Red or Brown queue
#SBATCH --mail-type=START,FAIL,END    # Send an email when the job finishes or fails

python src/baseline_.py