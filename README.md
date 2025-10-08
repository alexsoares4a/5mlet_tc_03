# 🚀 Tech Challenge | Fase 3 – Previsão do Preço do Bitcoin com Machine Learning

> **Projeto Pós-Graduação em Machine Learning Engineering – FIAP**  
> **Aluno:** Alex Soares da Silva, RM354660  
> **Turma:** 5MLET  
> **Data:** Setembro de 2025

---

## 🎯 Objetivo

O objetivo deste projeto é desenvolver um modelo preditivo capaz de **prever o preço de fechamento diário do Bitcoin (BTC-USD)** utilizando técnicas de Machine Learning, com foco em séries temporais.

O projeto segue a metodologia **CRISP-DM**, abrangendo todas as etapas:
1. Coleta e limpeza de dados
2. Análise Exploratória de Dados (EDA)
3. Engenharia de atributos e variáveis exógenas
4. Treinamento e comparação de modelos (LSTM, AutoARIMA, Prophet)
5. Seleção do modelo campeão
6. Deploy em dashboard interativo (Streamlit)

O resultado final é um sistema funcional que pode ser usado como base para aplicações financeiras, decisões de investimento ou monitoramento de mercado.

---

## 🔧 Tecnologias Utilizadas

| Tecnologia | Finalidade |
|----------|------------|
| Python | Linguagem principal |
| Jupyter Notebook | Desenvolvimento e análise |
| CoinGecko API | Coleta de dados do Bitcoin |
| yfinance | Coleta do Ethereum (ETH-USD) e DXY |
| alternative.me API | Índice de Medo e Ganância (Fear & Greed Index) |
| TensorFlow/Keras | Modelo LSTM (Deep Learning) |
| statsforecast | Modelo AutoARIMA com exógenas |
| Prophet | Modelo aditivo de séries temporais |
| Scikit-learn | Pré-processamento e métricas |
| Streamlit | Dashboard interativo de previsão |
| Pandas, NumPy, Matplotlib, Seaborn, Plotly | Manipulação e visualização de dados |

---

## 📁 Estrutura do Repositório
```
5mlet_tc_03/
├── app/
│   └── dashboard.py             # Aplicação Streamlit
├── data/
│   └── btc_historical.parquet   # Dados históricos processados
├── models/
│   ├── lstm_bitcoin.keras       # Modelo LSTM treinado
│   └── scaler_bitcoin.pkl       # Scaler salvo para normalização
├── video/
│   └── previsao_bitcoin.mp4     # Vídeo Explicativo
├── 5mlet_tc_03.ipynb            # Notebook completo com EDA e modelagem
├── requirements.txt             # Dependências do projeto
└── README.md                    # Este arquivo
```

## ▶️ Como Executar

1. **Clone o repositório**
    ```bash
    git clone https://github.com/alexsoares4a/5mlet_tc_03.git
    cd 5mlet_tc_03
    ```

2.  **Crie e ative um ambiente virtual**
    ```bash
    python -m venv venv
    # No Linux/macOS:
    source venv/bin/activate

    # No Windows:
    .\venv\Scripts\activate
    ```

3. **Instale as dependências**
    ```bash
    pip install -r requirements.txt
    ```

4. **Execute o notebook (opcional)**
    ```bash
    jupyter notebook notebooks/bitcoin_analysis.ipynb
    ```

4. **Execute o dashboard Streamlit**
    ```bash
    streamlit run app/dashboard.py
    ```
    O dashboard será aberto no navegador (http://localhost:8501).
 

## ✅ Modelo Campeão: LSTM
O **LSTM** demonstrou superioridade ao capturar padrões não lineares e volatilidade do mercado de criptomoedas. 
 
## 📎 Recursos Úteis

*   **Site do Projeto:** [https://5mlet-bitcoin.streamlit.app/](https://5mlet-bitcoin.streamlit.app/)

*   **Vídeo Explicativo:** [https://github.com/alexsoares4a/5mlet_tc_03/video/previsao_bitcoin.mp4](https://github.com/alexsoares4a/5mlet_tc_03/blob/main/video/previsao_bitcoin.mp4)

---
     