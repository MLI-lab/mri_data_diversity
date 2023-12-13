import csv
import requests
import os
from tqdm import tqdm

# create folder
try:
    os.makedirs("./data")
except FileExistsError:
    pass

# download data
with open('download_url.csv', newline='') as csvfile:
    list_of_urls = csv.reader(csvfile, delimiter=' ')
    
    # Determine number of items
    header = next(list_of_urls) # skip header
    n_items = len(list(list_of_urls))

    # Reset the iterator to the beginning of the file
    csvfile.seek(0)
    next(list_of_urls)  # skip header
    
    for row in tqdm(list_of_urls, total=n_items):
        url = ', '.join(row)
        r = requests.get(url, allow_redirects=True)
        fname = url.rsplit('/', 1)[-1]
        open('./data/'+fname+'.h5', 'wb').write(r.content)