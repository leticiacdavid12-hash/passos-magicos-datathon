# Datathon POSTECH - Fase 5: Passos Mágicos

Projeto de análise de dados e predição de risco de defasagem educacional para a Associação Passos Mágicos, desenvolvido como trabalho de conclusão de curso da Pós-Graduação em Data Analytics da POSTECH.

## Estrutura do Projeto

passos-magicos-datathon/
├── data/
│   ├── raw/                        # Dataset original (não commitado)
│   └── processed/                  # Parquets limpos
├── notebooks/
│   ├── 01_eda.ipynb                # Limpeza e EDA
│   ├── 02_business_questions.ipynb # Perguntas de negócio
│   └── 03_model.ipynb              # Modelo preditivo
├── app/
│   └── streamlit_app.py            # Aplicação Streamlit
├── models/                         # Artefatos do modelo (.pkl)
├── reports/                        # Gráficos exportados
├── requirements.txt
└── README.md

## Como Executar Localmente

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Deploy

Acesse a aplicação em produção:
[link]

## Modelo Preditivo

- **Algoritmo**: Regressão Logística  
- **AUC-ROC**: 0,9976  
- **F1-Score**: 0,9542  
- **Threshold**: 0,441  
- **Features**: 15 (sem INDE nem PEDRA — evitar data leakage)
