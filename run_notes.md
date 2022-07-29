### Update operating system:
`sudo apt update && sudo apt upgrade`

and restart

### Install miniconda
```
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
bash Mambaforge-Linux-x86_64.sh
```
* must confirm license and location (use defaults)
* close and reopen shell

### Intall git
```
sudo apt-get install git
```

Note: this is probably already installed, in which case this is just a check.

### "Install" the `gfcat` code
```
git clone https://github.com/MillionConcepts/gfcat.git
cd gfcat
mamba create -n gfcat python=3.8 -y
mamba activate gfcat
```

### Install the dependencies
```
mamba install scipy -y
mamba install ipython -y
mamba install tqdm -y
mamba install sqlalchemy -y
mamba install numpy -y
mamba install pandas -y
mamba install scikit-learn -y
mamba install matplotlib -y
mamba install astropy -y
mamba install rich -y
mamba install pyarrow -y
mamba install astroquery -y 
```


### Install AWS CLI
```
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt install unzip
```
```
unzip awscliv2.zip
sudo ./aws/install
aws configure
```
and then enter the appropriate AWS credentials.


### Mount the EBS volume
```
lsblk
```
find the 100Gb EBS volume... maybe xvdb... maybe nvme1n1... etc. and sbu that in below
```
sudo file -s /dev/xvdb
sudo mkfs -t ext4 /dev/xvdb
cd
sudo mkdir datadir
sudo mount /dev/xvdb datadir
sudo chown -R ubuntu:ubuntu datadir
```

### Create the data paths
```
cd datadir
rm -rf lost+found/
mkdir photom
mkdir plots
```

### Verify that the paths in `make_gfcat.py` are appropriate
```
cd ~/gfcat
vi make_gfcat.py
```

### Make GFCAT
```
python make_gfcat.py
```
