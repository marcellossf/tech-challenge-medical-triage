# Model Card - Medical Abstracts Classifier 0.1.0

## Visão geral

Classificador de textos médicos em inglês com TF-IDF e Logistic Regression. Existe uma
versão nativa Scikit-Learn e uma conversão ONNX Runtime com gate de paridade.

## Uso pretendido

- Demonstração acadêmica de ciclo de vida de modelo.
- Comparação de runtime e latência.
- Exercício de API, CI/CD, orquestração e observabilidade.

## Fora de escopo

- Diagnóstico, recomendação clínica ou decisão de emergência.
- Uso com dados reais de paciente sem governança, base legal e validação.
- Generalização para português ou para instituições diferentes.
- Interpretar a regra `high/medium/low` como protocolo de triagem validado.

## Dados

- Fonte: [Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus).
- Versão fixada: commit `70a2d9106c724729be8b3c4ddb00d1b14ec300c8`.
- Licença dos dados: CC BY-SA 3.0.
- Treino: 11.550 registros.
- Teste oficial: 2.888 registros.
- Idioma: inglês.
- Classes: cinco grupos de condições médicas.

Os scripts validam esquema, valores ausentes, labels e checksums. O split oficial é
preservado.

## Modelo

- `TfidfVectorizer`, unigramas, até 40.000 features.
- `LogisticRegression` com `class_weight="balanced"`.
- Seed 42.
- ONNX opset 17.

O TF-IDF linear e a tokenização ASCII foram escolhidos porque `sublinear_tf=True`
introduziu divergência no conversor ONNX. A configuração final passou no gate de
paridade em todo o conjunto de teste.

## Resultados

| Métrica | Valor |
|---|---:|
| Accuracy | 0,5918 |
| F1 macro | 0,5951 |
| Recall macro | 0,6364 |
| Paridade de classe ONNX | 1,0000 |
| Diferença máxima de probabilidade | 1,96e-7 |

O menor F1 por classe foi observado em `general pathological conditions` (0,4231),
uma categoria ampla e mais difícil. Os resultados demonstram o pipeline, mas não são
evidência de desempenho clínico.

## Latência

300 inferências single-record após 30 warm-ups por runtime:

| Percentil | Scikit-Learn | ONNX | Ganho |
|---|---:|---:|---:|
| p50 | 0,3778 ms | 0,1538 ms | 2,46x |
| p95 | 0,5781 ms | 0,1991 ms | 2,90x |
| p99 | 0,7073 ms | 0,2309 ms | 3,06x |

Os valores dependem do hardware, carga, runtime e sistema operacional. Reproduza o
benchmark no ambiente usado na apresentação.

## Prioridade demonstrativa

A condição prevista não determina urgência. A API aplica uma lista explícita de termos
de alerta e devolve `priority_source="business_rule"` com uma justificativa. Essa regra
é apenas didática, pode produzir falsos positivos/negativos e não foi revisada como
protocolo clínico.

## Riscos e melhorias

- Avaliar dataset com urgência anotada por profissionais.
- Criar validação externa e análise de vieses.
- Calibrar probabilidades.
- Testar robustez a negação, abreviações e mudança de idioma.
- Implantar revisão humana e política de fallback.
- Monitorar drift, distribuição de confiança e erros sem registrar PHI.
