# Decisão arquitetural

## Resumo

O sistema separa inferência real-time de treinamento batch. A API permanece pequena,
sem Airflow no caminho crítico, enquanto a DAG produz e valida novos artefatos de forma
assíncrona.

```mermaid
flowchart LR
    C[Cliente] --> A[FastAPI]
    A --> P[Pipeline TF-IDF]
    P --> S[Scikit-Learn]
    P --> O[ONNX Runtime]
    A --> M[/metrics]
    M --> PR[Prometheus]
    PR --> G[Grafana]

    D[Dataset] --> AF[Airflow]
    AF --> T[Treino]
    T --> E[Avaliação]
    E --> X[Exportação ONNX]
    X --> AR[Artefatos versionados]
```

## Local

- A API carrega o artefato uma única vez no startup.
- O backend é selecionado por configuração (`sklearn` ou `onnx`).
- Prometheus coleta métricas na própria API.
- Grafana é provisionado por arquivos, sem configuração manual.
- O Airflow usa uma stack separada para não aumentar o tempo de inicialização da demo.

## Nuvem escolhida: AWS

A implantação recomendada usa AWS ECS Fargate para o serviço real-time:

- Amazon ECR armazena a imagem.
- Application Load Balancer distribui requisições e executa health checks.
- ECS Fargate remove a necessidade de administrar servidores.
- Auto Scaling responde a volume de requisições e utilização.
- S3 armazena modelos, metadados e relatórios de benchmark.
- GitHub Actions autentica por OIDC, sem chave AWS permanente no repositório.
- Amazon MWAA executa o retreino batch quando a complexidade justificar Airflow gerenciado.
- Amazon Managed Service for Prometheus e Amazon Managed Grafana preservam a estratégia
  de observabilidade usada localmente.

## Batch versus real-time

Inferência é real-time porque o valor da triagem depende de resposta imediata. Treino,
avaliação e promoção de modelo são batch porque consomem mais recursos, não pertencem
ao caminho da requisição e precisam de validações antes de publicar um artefato.

## Segurança e limites clínicos

- O protótipo não substitui avaliação médica nem deve orientar emergência real.
- Texto de entrada não aparece em labels, logs estruturados ou métricas Prometheus.
- Segredos e dados brutos não são versionados.
- A API deve ficar atrás de TLS, autenticação e limitação de taxa em produção.
- A prioridade é uma regra acadêmica separada do classificador de condição e não possui
  validação clínica.
