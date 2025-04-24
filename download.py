import os
import shutil
import requests
import tarfile

# HOME = "/home/bourse/data/"   # we expect subdirectories boursorama and euronext
HOME="./data/" # for local testing

# Create data directory if it doesn't exist
os.makedirs(HOME, exist_ok=True)

# Create and clean bourso directory
os.makedirs(os.path.join(HOME, 'bourso/'), exist_ok=True)
for f in os.listdir(os.path.join(HOME, 'bourso/')):
    if f.startswith('20'):
        shutil.rmtree(os.path.join(HOME, 'bourso/', f))

# Download and extract bourso data
stream = requests.get('https://www.lrde.epita.fr/~ricou/pybd/projet/bourso.tgz', stream=True)
tarfile.open(fileobj=stream.raw, mode='r|gz').extractall(os.path.join(HOME, 'bourso/')) # try 'r:gz' if there is an error

# Create and clean euronext directory 
os.makedirs(os.path.join(HOME, 'euronext/'), exist_ok=True)
if os.path.exists(os.path.join(HOME, 'euronext/')):
    shutil.rmtree(os.path.join(HOME, 'euronext/'))
os.makedirs(os.path.join(HOME, 'euronext/'))

# Download and extract euronext data
stream = requests.get('https://www.lrde.epita.fr/~ricou/pybd/projet/euronext.tgz', stream=True)
tarfile.open(fileobj=stream.raw, mode='r|gz').extractall(os.path.join(HOME, 'euronext/')) # try 'r:gz' if there is an error