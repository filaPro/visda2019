import os
from argparse import ArgumentParser

from utils import get_track_information

BASE_URL = 'http://csr.bu.edu/ftp/visda/2019/'


def download(path, url):
    url = f'{BASE_URL}{url}'
    print(f'downloading: {url}')
    os.system(f'wget -O {path} {url}')


def extract(path):
    print(f'extracting: {path}')
    destination_name = f'{path[:-4]}_'
    os.system(f'unzip {path} -d {destination_name}')
    name = os.listdir(destination_name)[0]
    os.system(f'mv {os.path.join(destination_name, name)} {path[:-4]}')
    os.system(f'rm -r {destination_name}')


def download_track(path, track):
    phases, domains, name = get_track_information(track)
    path = os.path.join(path, name, 'raw')
    os.makedirs(path, exist_ok=True)
    if len(os.listdir(path)) != 0:
        raise ValueError('Output directory is not empty')
    for domain in domains:
        url_name = name.replace('_', '-')
        download(os.path.join(path, f'{domain}.zip'), f'{url_name}/{domain}.zip')
        extract(os.path.join(path, f'{domain}.zip'))
        for phase in phases:
            phase_name = phase if phase != 'unlabeled' else 'unl'
            download(os.path.join(path, f'{domain}_{phase}.txt'), f'{url_name}/txt/{domain}_{phase_name}.txt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='/content/data')
    options = vars(parser.parse_args())

    for i in (0, 1):
        download_track(path=options['path'], track=i)
