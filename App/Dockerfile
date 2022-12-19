FROM python:3
RUN pip install flask
RUN pip install requests
RUN pip install gcloud
RUN pip install --upgrade google-cloud-translate
EXPOSE 5003/tcp
COPY app.py .
COPY fmartinezlopezGTranslateKey.json .
COPY templates/index.html templates/
ENTRYPOINT [ "python3", "app.py" ]