# Приложение для улучшения качества лица на видео с помощью DFDNet сети на основе istio #
<br><br>
### САНКТ-ПЕТЕРБУРГСКИЙ ГОСУДАРСТВЕННЫЙ УНИВЕРСИТЕТ<br>
### Направление: 02.03.02 «Фундаментальная информатика и информационные технологии<br>
### ООП: Программирование и информационные технологии
<br><br>

Работу выполнили студенты 4 курса:<br>
Князев Никита Андреевич 431 группа<br>
Шарафутдинов Тимур Рустемович 433 группа
<br><br>
<h3>ТЗ:</h3>
Сервис параллельной обработки видео файлов пользователя группой (парой нейронных сетей или выше) работающий под контролем Istio<br><br>
<ol>
  <li>найти нейронную сеть для обработки видео, например улучшение качества кадров</li>
  <li>запустить сеть локально</li>
  <li>создать контейнер с сетью в Docker</li>
  <li>сделать yml файл для docker-compose и istio</li>
</ol>
<br><br>
Шаги 1-3 выполнены успешно, однако загрузить все сеть в контейнер не вышло, тк для работы нужно GPU, а nvidia-docker-compose не поддерживает windows (лок. хост проекта), поэтому в контейнер была загружена только часть сети. Использование CPU в контейнере также имело некоторые проблемы.
<br><br>
<h3>Перед запуском</h3>
Для запуска сети необходимо в папке DFDNet создать папку weights и скачать туда веса модели (https://yadi.sk/d/PWTw92J5PAzMZg)<br>
Также нужно скачать модели распознования в папку DictionaryCenter512 (https://yadi.sk/d/mC8U2zWDvFW6Yg)
<br>
<h3>Требования:</h3>
<ul>
  <li>Nvidia GPU</li>
  <li>Docker Desktop</li>
  <li>Python 3.7+(для локального запуска)</li>
</ul>
<br>
<h3>Запуск:</h3>
<ol>
  <li>Скачать репозиторий (git clone) и веса, модели</li>
  <li>Открыть докер, в корне запустить консоль и выполнить docker-compose up (или index.py для локального запуска на Flask)</li>
  <li>Открыть http://localhost:4000/ (5000 для Flask) в браузере</li>
</ol>

<h3>Демо</h3>
https://youtu.be/oopfX5Dx-Nc
