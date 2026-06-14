FROM python:3.13-slim

RUN useradd -m -s /bin/bash agent

WORKDIR /root/task

COPY requirements.txt .
RUN pip install --quiet --no-cache-dir -r requirements.txt

COPY . .

RUN chmod -R 755 /root/task

CMD ["/bin/bash"]
