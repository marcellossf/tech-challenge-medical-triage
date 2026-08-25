FROM apache/airflow:2.10.5-python3.11

USER root

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes locales \
    && sed -i 's/^# *\(en_US.UTF-8 UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

USER airflow

COPY pyproject.toml README.md /opt/airflow/project/
COPY src /opt/airflow/project/src
COPY training /opt/airflow/project/training

RUN pip install --no-cache-dir '/opt/airflow/project[pipeline]'

WORKDIR /opt/airflow/project
