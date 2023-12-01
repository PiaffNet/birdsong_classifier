FROM python:3.10-slim


# Indication du working directory
WORKDIR /prod

# on copie les requirements et on les install
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt


# on copie les fichiers dont on a besoin pour construire l'image API
COPY birdsong birdsong
COPY Makefile Makefile


#CMD (API à faire, juste pour l'exemple)
CMD uvicorn bird.api.fast:app --host 0.0.0.0 --port $PORT
