FROM registry.access.redhat.com/ubi9/python-39

ADD requirements.txt .
RUN pip install -r requirements.txt
ADD . .

ENV ODRS_HOME=/opt/app-root/src \
    LANG=C \
    APP_MODULE=odrs:app

CMD /opt/app-root/src/entrypoint
