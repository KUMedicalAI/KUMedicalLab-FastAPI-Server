FROM python:3.8.8
RUN pip install fastapi uvicorn tensorflow python-multipart numpy
COPY ./start.sh /start.sh
RUN chmod +x /start.sh
COPY . .
CMD ["ls"]
CMD ["/bin/bash /start.sh"]
