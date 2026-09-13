# Checklist da rubrica

| Critério | Peso | Evidência esperada | Status |
|---|---:|---|---|
| Modelagem e otimização | 20% | modelo Scikit-Learn, ONNX, paridade e benchmark | Validado localmente |
| CI/CD | 15% | lint, testes e build no GitHub Actions | Validado no commit final |
| Airflow | 15% | DAG de preparo, treino, exportação e benchmark | Run completo validado em Docker |
| Monitoramento | 20% | Compose, Prometheus e dashboard Grafana provisionado | Stack validada com tráfego real |
| README | 15% | comandos reproduzíveis e decisão AWS | Concluído |
| Vídeo STAR | 15% | link com duração máxima de cinco minutos | Pendente de gravação |

## Evidências antes da entrega

- [x] CI verde no commit final.
- [x] DAG concluída disponível no Airflow local.
- [x] Prometheus com target da API em estado `UP`.
- [x] Dashboard Grafana validado com tráfego real.
- [x] JSON do dashboard versionado.
- [x] Benchmark executado no ambiente da apresentação.
- [ ] Link do vídeo testado em janela anônima.
- [ ] README reproduzido a partir de um clone limpo.
