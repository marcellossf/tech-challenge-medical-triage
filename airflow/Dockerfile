FROM apache/airflow:2.10.5-python3.11

COPY pyproject.toml README.md /opt/airflow/project/
COPY src /opt/airflow/project/src
COPY training /opt/airflow/project/training

RUN pip install --no-cache-dir /opt/airflow/project

WORKDIR /opt/airflow/project
