# Checklist da rubrica

| Critério | Peso | Evidência esperada | Status |
|---|---:|---|---|
| Modelagem e otimização | 20% | modelo Scikit-Learn, ONNX, paridade e benchmark | Validado localmente |
| CI/CD | 15% | lint, testes e build no GitHub Actions | Configurado; execução remota pendente |
| Airflow | 15% | DAG de preparo, treino, exportação e benchmark | Validado estaticamente; execução Docker pendente |
| Monitoramento | 20% | Compose, Prometheus e dashboard Grafana provisionado | Validado estaticamente; execução Docker pendente |
| README | 15% | comandos reproduzíveis e decisão AWS | Concluído |
| Vídeo STAR | 15% | link com duração máxima de cinco minutos | Pendente de gravação |

## Evidências antes da entrega

- [ ] CI verde no commit final.
- [ ] Print da DAG concluída.
- [ ] Prometheus com target da API em estado `UP`.
- [ ] Dashboard Grafana com tráfego real nos painéis.
- [ ] JSON do dashboard versionado.
- [ ] Benchmark executado no ambiente da apresentação.
- [ ] Link do vídeo testado em janela anônima.
- [ ] README reproduzido a partir de um clone limpo.
