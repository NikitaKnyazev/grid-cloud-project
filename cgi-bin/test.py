import os

url = "https://www.youtube.com/watch?v=si-thUvEvls"
time1 = "00:00:34"
time2 = "00:00:40"

file_path="DFDNet/test_FaceDict.py --gpu_ids -1 --source_url "+url+" --start "+time1+" --stop "+time2

os.system(f'py {file_path}')
print('Done')
