import pandas as pd
import os
import requests
from tqdm import tqdm

# create folder
try:
    os.makedirs("./data")
except FileExistsError:
    pass


df = pd.read_csv('download_files.csv')
url = df['Url']
name = df['name']
for u, n in tqdm(zip(url, name), total=len(name)):
    u = u[39:-12]
    u = 'https://dataverse.nl/api/access/datafile/' + u + '?gbrecs=true'
    r = requests.get(u, allow_redirects=True)
    open('./data/'+n, 'wb').write(r.content)