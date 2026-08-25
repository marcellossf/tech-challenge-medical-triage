# Dados

O dataset não é versionado neste repositório. O pipeline baixa o
[Medical Abstracts TC Corpus](https://github.com/sebischair/Medical-Abstracts-TC-Corpus),
distribuído sob CC BY-SA 3.0, e registra a origem e o checksum no metadata do modelo.

Os rótulos originais representam cinco condições médicas. A prioridade retornada pela
API é uma regra de negócio separada e é identificada explicitamente como tal; ela não
é apresentada como urgência aprendida ou decisão clínica validada.

Versão usada:

- commit upstream: `70a2d9106c724729be8b3c4ddb00d1b14ec300c8`
- treino: 11.550 registros
- teste: 2.888 registros
- idioma: inglês
- classes: cinco grupos de condições médicas
