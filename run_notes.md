### Update operating system:
`sudo apt update && sudo apt upgrade`
and restart

### Install miniconda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
* must confirm license and location (use defaults)
* close and reopen shell

### Intall git
`sudo apt-get install git`

Note: this is probably already installed, in which case this is just a check.

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

### "Install" the `gfcat` code
```
git clone https://github.com/MillionConcepts/gfcat.git
cd gfcat
conda create -n gfcat python=3.8 -y
conda activate gfcat
```

### Install the dependencies
```
conda install -c anaconda ipython -y
conda install -c conda-forge tqdm -y
conda install -c anaconda sqlalchemy -y
conda install -c anaconda numpy -y
conda install -c anaconda pandas -y
conda install -c anaconda scipy -y
conda install -c anaconda scikit-learn -y
conda install -c anaconda matplotlib -y
conda install -c anaconda astropy -y
conda install -c conda-forge pyarrow -y
```

### Mount the EBS volume
```
lsblk
```
find the 100Gb EBS volume... maybe xvdb... maybe nvme1n1... etc. and sbu that in below
```
sudo file -s /dev/nvme1n1
sudo mkfs -t ext4 /dev/nvme1n1
cd
sudo mkdir datadir
sudo mount /dev/nvme1n1 datadir
sudo chown -R ubuntu:ubuntu datadir
```

### Verify that the paths in `make_gfcat.py` are appropriate
```
cd gfcat
vi make_gfcat.py
```

### Make GFCAT
`python make_gfcat.py`
