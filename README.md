# Submission Details
Please ignore the outfiles and the bash scripts.

To take a closer look at our implementation, these are the main directories/files to look at:

- /config
- /models
- /scheduler
- train_cifar10.py
- generate_cifar10.py
- train_mnist.py
- generate_mnist.py


# Scholar Instructions

## cloning repo
- create ssh key using `ssh-keygen`
- add ssh key to github
- clone git repo using ssh instead of http

## venv
- create venv using `python3 -m venv .venv`
- activate venv using `source .venv/bin/activate`
- install necessary libraries using pip (install torch should be enough)

## running
- run `sbatch train_mnist.sh`
- check status using `squeue -u $USER`
- check logs by opening the file named `slurm-123124124134123` that gets generated in the directory
