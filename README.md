#### Google Colab installation
```
!git clone https://github.com/filaPro/visda2019 /tmp/visda2019
!cp -rf /tmp/visda2019/* .
!rm -r /tmp/visda2019
!rm -r sample_data
!pip install -r requirements.txt

from google.colab import drive
data_path = '/content/data/'
!mkdir $data_path
drive_data_path = data_path + 'drive'
!mkdir $drive_data_path
drive.mount(drive_data_path)
raw_data_path = data_path + 'raw'
!mkdir $raw_data_path
!cp $drive_data_path/My\ Drive/visda2019/* $raw_data_path
from utils import unzip_raw_data, DOMAINS
unzip_raw_data(raw_data_path, DOMAINS)
```

#### ML Platrfom installation
```
sudo apt-get update
sudo apt-get install htop vim screen git
git clone https://github.com/filaPro/visda2019
pip3 install --user -r visda2019/requirements.txt
export PYTHONPATH=$PYTHONPATH:/home/user/visda2019
```
