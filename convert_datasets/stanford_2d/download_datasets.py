import csv
import requests
import os

# create folder
try:
    os.makedirs("./data")
except FileExistsError:
    pass

# download data
with open('download_url.csv', newline='') as csvfile:
    list_of_urls = csv.reader(csvfile, delimiter=' ')
    next(list_of_urls) # skip header
    for row in list_of_urls:
        url = ', '.join(row)
        r = requests.get(url, allow_redirects=True)
        fname = url.rsplit('/', 1)[-1]
        open('./data/'+fname+'.h5', 'wb').write(r.content)