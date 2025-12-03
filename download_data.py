import wfdb
import os

data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# MIT-BIH 100-as record - - download
wfdb.dl_database('mitdb', dl_dir=data_dir, records=['100'])

print("Download completed. Files are in the 'data/' folder.")

