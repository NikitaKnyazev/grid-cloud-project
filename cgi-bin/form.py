import os, cgi
import sys
import subprocess

# Получаем GET параметр, переданный из HTML формы
form = cgi.FieldStorage()
url = form.getfirst("url", "https://www.youtube.com/watch?v=si-thUvEvls")
time1 = form.getfirst("timecod1", "00:00:00")
time2 = form.getfirst("timecod2", "00:00:00")

# Открываем HTML шаблон странички с результатами
f=open('template/form.html','r')
htmltemplate=f.read()
f.close()

# Выводим в браузер
print("Content-type: text/html\n")
print(htmltemplate)

# Запускаем внешний скрипт и передаем ему
#test_FaceDict.py --test_path ./TestData/TestWhole --results_dir ./Results/TestWholeResults --upscale_factor 4 --gpu_ids -1
#file_path="cgi-bin/DFDNet/test_FaceDict.py --gpu_ids -1 --source_url "+url+" --start "+time1+" --stop "+time2
file_path="test_FaceDict.py --gpu_ids -1 --source_url "+url+" --start "+time1+" --stop "+time2
os.system(f'py {file_path}')

# Открываем HTML шаблон странички с результатами
f=open('template/form2.html','r')
htmltemplate=f.read()
f.close()

# Выводим в браузер
#print("Content-type: text/html\n")
print(htmltemplate)

subprocess.call("template/video.html", shell=True)
