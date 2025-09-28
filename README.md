# 🚀 Tech Challenge | Fase 3 – Previsão do Preço do Bitcoin com Machine Learning

> **Projeto Final da Pós-Graduação em Machine Learning Engineering – FIAP**  
> **Aluno:** Alex Soares da Silva, RM354660  
> **Turma:** 5MLET_TC_03  
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
tech-challenge-fase3/
├── notebooks/
│   └── bitcoin_analysis.ipynb        # Notebook completo com EDA e modelagem
├── app/
│   └── dashboard.py                  # Aplicação Streamlit
├── models/
│   ├── lstm_bitcoin.keras          # Modelo LSTM treinado
│   └── scaler_bitcoin.pkl          # Scaler salvo para normalização
├── data/
│   └── btc_historical.parquet      # Dados históricos processados
├── requirements.txt                # Dependências do projeto
└── README.md                       # Este arquivo



---

## ▶️ Como Executar

### 1. Clone o repositório
```bash
git clone https://github.com/seuusuario/tech-challenge-fase3.git
cd tech-challenge-fase3

2. Instale as dependências
pip install -r requirements.txt

3. Execute o notebook (opcional)
jupyter notebook notebooks/bitcoin_analysis.ipynb

4. Execute o dashboard Streamlit
streamlit run app/dashboard.py

O dashboard será aberto no navegador (http://localhost:8501).

📊 Resultados dos Modelos 

Após avaliação rigorosa, os modelos foram comparados com base nas métricas: 
LSTM
	
839,37
	
990,09
	
0,878
AutoARIMA (com exógenas)
	
3.876,50
	
4.195,14
	
-1,492
Prophet (com múltiplas exógenas)
	
5.651,84
	
5.855,57
	
-3,855
 
 

✅ Modelo Campeão: LSTM
Com R² de 87,8%, o LSTM demonstrou superioridade ao capturar padrões não lineares e volatilidade do mercado de criptomoedas. 
 
📹 Vídeo Explicativo 

Assista ao storytelling completo do projeto, onde explico: 

    O problema e fontes de dados
    Análise exploratória
    Comparação de modelos
    Escolha do LSTM como campeão
    Funcionamento do dashboard
     

🔗 Assistir ao vídeo no YouTube  
 
💡 Insights e Conclusões 

    O Fear & Greed Index apresentou alta correlação positiva (0,55) com o BTC, indicando que períodos de "ganância" tendem a coincidir com altas.
    O Índice Dólar (DXY) também teve correlação positiva (+0,32), contrariando a expectativa de relação negativa — possível sinal de alinhamento macroeconômico global.
    Modelos lineares (AutoARIMA) e aditivos (Prophet) falharam em cenários de alta volatilidade, enquanto o LSTM se destacou por sua robustez.
     

 
🛠️ Melhorias Futuras 

    Incluir mais variáveis exógenas: notícias, volume de busca (Google Trends), stablecoins.
    Implementar atualização automática dos dados via script agendado.
    Expandir o dashboard com previsões em tempo real e alertas.
    Testar modelos híbridos (LSTM + Prophet) para melhorar acurácia.
     
     

 
📚 Referências 

    CoinGecko API: https://www.coingecko.com/en/api 
    Alternative.me Fear & Greed Index: https://alternative.me/crypto/fear-and-greed-index/ 
    Yahoo Finance (yfinance): https://pypi.org/project/yfinance/ 
    StatsForecast: https://nixtla.io/statsforecast/ 
    Prophet: https://facebook.github.io/prophet/ 
    TensorFlow: https://www.tensorflow.org/ 
     