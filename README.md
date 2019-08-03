# Google Colab installation
```
!git clone https://github.com/filaPro/visda2019 /tmp/visda2019
!cr -r /tmp/visda2019/* .
!rm -r /tmp/visda
!rm -r sample_data
!pip install -r requirements.txt

from google.colab import drive
drive.mount('/content/data/drive')
raw_data_path = '/content/data/raw'
!cp /content/data/drive/My Drive/visda2019/* $raw_data_path
from utils import unzip_raw_data
unzip_raw_data(raw_data_path)
```
