FROM python:3.8

USER 10014

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade --user -r /code/requirements.txt

# 
COPY . /code/app

# 
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]
