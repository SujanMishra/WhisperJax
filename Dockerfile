FROM python:3.10-slim
# alias
RUN echo 'alias python="python3"' >> ~/.bashrc
RUN echo 'alias pip="pip3"' >> ~/.bashrc



WORKDIR /

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

COPY src ./src

WORKDIR /src

RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


WORKDIR /src/whisper-jax
RUN pip install -e .["endpoint"]
RUN pip install --upgrade gradio_client

ENV WORKSPACE="/workspace"
ENV PERSISTENT_FOLDER="${WORKSPACE}/persistent"
WORKDIR ${WORKSPACE}
VOLUME [ "${PERSISTENT_FOLDER}", "${WORKSPACE}/scratch" ]
ARG PORT=7860
EXPOSE PORT
ARG WHISPER_APP_PATH=./src/whisper-jax/app/app.py

CMD ["python","${WHISPER_APP_PATH}", "--host", "0.0.0.0", "--port", "${PORT}", "--reload"]
