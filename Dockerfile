# syntax=docker/dockerfile:1
FROM tensorflow/tensorflow:latest-gpu
WORKDIR ./
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8888
COPY . .
# CMD ["jupyter", "notebook", "--allow-root"]
CMD ["python", "train_model.py"]