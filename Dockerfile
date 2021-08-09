FROM python:3.8
LABEL maintainer="stijn.rosaer@telenet.be"

# Template config
ENV APP_ENTRYPOINT web
ENV LOG_LEVEL info
ENV MU_SPARQL_ENDPOINT 'http://database:8890/sparql'
ENV MU_SPARQL_UPDATEPOINT 'http://database:8890/sparql'
ENV MU_APPLICATION_GRAPH 'http://mu.semte.ch/application'

WORKDIR /app
ADD ./requirements.txt /app/
RUN pip3 install -r requirements.txt

ADD . /app/

CMD ["python", "-u", "web.py"]