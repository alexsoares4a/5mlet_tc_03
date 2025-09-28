import streamlit as st
from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

# Configurar título e ícone da página
st.set_page_config(
    page_title="🔮 Previsão do Preço do Bitcoin",
    page_icon="₿",
    layout="wide"
)

# Título do app
st.title("📊 Dashboard de Previsão do Preço do Bitcoin (BTC-USD)")
st.markdown("Modelo: **LSTM** | Fonte: CoinGecko API | Atualizado diariamente")

# --- Funções Auxiliares ---
@st.cache_data
def carregar_dados():
    """Carrega dados históricos do Bitcoin com exógenas"""
    if os.path.exists("data/btc_historical.parquet"):
        df = pd.read_parquet("data/btc_historical.parquet")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        # Renomear para padronizar
        df = df.rename(columns={'btc_close': 'Close'})
        return df[['Close', 'fgi', 'eth_close', 'dxy']]
    else:
        st.error("Dados não encontrados. Execute o coletor primeiro.")
        return None

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo LSTM treinado"""
    try:
        model = load_model('models/lstm_bitcoin.keras')
        st.success("✅ Modelo LSTM carregado com sucesso!")
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        return None

def preparar_entrada(dados, scaler, seq_length=20):
    """Prepara a última sequência de dados para previsão"""
    ultimos_dados = dados['Close'].tail(seq_length).values.reshape(-1, 1)
    dados_normalizados = scaler.transform(ultimos_dados)
    X = np.reshape(dados_normalizados, (1, seq_length, 1))
    return X

def gerar_previsoes(modelo, scaler, dados, num_dias=7):
    """
    Gera previsões para os próximos dias.
    - modelo: modelo LSTM carregado
    - scaler: MinMaxScaler ajustado nos dados
    - dados: DataFrame com colunas ['Close', 'fgi', 'eth_close', 'dxy']
    - num_dias: número de dias à frente
    """
    seq_length = 20
    n_features = 4  # btc_close, fgi, eth_close, dxy

    # Extrair os últimos 'seq_length' dias de todas as features
    ultimos_dados = dados[['Close', 'fgi', 'eth_close', 'dxy']].tail(seq_length).values
    entrada = scaler.transform(ultimos_dados)  # Normalizar

    previsoes = []
    datas = []

    ultimo_dia = dados.index[-1]
    current_date = ultimo_dia + timedelta(days=1)

    while len(previsoes) < num_dias:
        # Garantir que entrada tenha shape (1, 20, 4)
        X = entrada.reshape(1, seq_length, n_features)
        
        # Fazer previsão (só prevemos o Close)
        pred = modelo.predict(X, verbose=0)[0, 0]

        # Armazenar previsão (só o Close)
        previsoes.append(pred)
        datas.append(current_date)

        # Atualizar entrada: remover primeiro dia, adicionar novo dia
        # Para as exógenas, usamos os últimos valores conhecidos (ou projeções simples)
        novo_dia = np.array([
            pred,  # Close previsto (será normalizado depois)
            entrada[-1, 1],  # fgi: último valor
            entrada[-1, 2],  # eth_close: último valor
            entrada[-1, 3]   # dxy: último valor
        ])
        
        entrada = np.vstack([entrada[1:], novo_dia])

        current_date += timedelta(days=1)

    # Reverter escala das previsões (Close)
    dummy = np.zeros((len(previsoes), 3))  # 3 outras features
    previsoes_escaladas = scaler.inverse_transform(np.hstack([np.array(previsoes).reshape(-1, 1), dummy]))[:, 0]
    
    return datas, previsoes_escaladas

# --- Carregar recursos ---
df = carregar_dados()
modelo = carregar_modelo()

if df is None or modelo is None:
    st.stop()

# Inicializar scaler com os dados
from joblib import load
scaler = load('models/scaler_bitcoin.pkl')
st.success("✅ Scaler carregado com 4 features")

# --- Sidebar: Entrada do usuário ---
st.sidebar.header("⚙️ Configurações")
num_dias = st.sidebar.slider("Dias à frente:", min_value=1, max_value=30, value=7)
fazer_previsao = st.sidebar.button("🚀 Gerar Previsão")

# --- Abas ---
tab1, tab2, tab3 = st.tabs(["📈 Previsão", "🔍 Dados Históricos", "ℹ️ Sobre"])

# --- Aba 1: Previsão ---
with tab1:
    st.subheader("🔮 Previsão Futura do Bitcoin")

    if fazer_previsao:
        with st.spinner("Gerando previsões..."):
            try:
                datas_futuras, previsoes = gerar_previsoes(modelo, scaler, df, num_dias)

                # Criar DataFrame de previsões
                df_pred = pd.DataFrame({
                    'Data': datas_futuras,
                    'Previsão (USD)': previsoes
                })

                # Mostrar tabela
                st.write("### 📅 Previsões")
                st.dataframe(df_pred.style.format({"Previsão (USD)": "US$ {:,.2f}"}))

                # Gráfico com Plotly
                fig = go.Figure()

                # Histórico
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Close'],
                    mode='lines', name='Histórico',
                    line=dict(color='blue')
                ))

                # Previsões
                fig.add_trace(go.Scatter(
                    x=datas_futuras, y=previsoes,
                    mode='lines+markers', name='Previsão',
                    line=dict(color='orange', dash='dot'),
                    marker=dict(size=6)
                ))

                fig.update_layout(
                    title='Preço do Bitcoin: Histórico e Previsão',
                    xaxis_title='',
                    yaxis_title='Preço (USD)',
                    hovermode='x unified',
                    legend=dict(x=0, y=1)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Destaque da próxima previsão
                proxima = previsoes[0]
                st.success(f"**Próximo fechamento estimado: US$ {proxima:,.2f}**")

            except Exception as e:
                st.error(f"Erro na geração de previsão: {e}")

    else:
        st.info("👈 Ajuste as configurações na barra lateral e clique em 'Gerar Previsão'.")

# --- Aba 2: Dados Históricos ---
with tab2:
    st.subheader("📊 Dados Históricos do Bitcoin")
    st.write(f"Período: {df.index[0].strftime('%d/%m/%Y')} até {df.index[-1].strftime('%d/%m/%Y')}")
    st.write(f"Total de registros: {len(df)}")

    # Estatísticas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Atual", f"US$ {df['Close'].iloc[-1]:,.2f}")
    col2.metric("Média", f"US$ {df['Close'].mean():,.2f}")
    col3.metric("Mínimo", f"US$ {df['Close'].min():,.2f}")
    col4.metric("Máximo", f"US$ {df['Close'].max():,.2f}")

    # Gráfico de histórico
    st.line_chart(df['Close'])

# --- Aba 3: Sobre ---
with tab3:
    st.subheader("ℹ️ Sobre este Projeto")
    st.write("""
    Este dashboard faz parte do **Tech Challenge da FIAP - Pós em Machine Learning Engineering**.
    
    **Objetivo**: Prever o preço de fechamento diário do Bitcoin usando um modelo LSTM.
    
    **Tecnologias utilizadas**:
    - `yfinance` / `CoinGecko API`: Coleta de dados
    - `TensorFlow/Keras`: Treinamento do modelo LSTM
    - `Streamlit`: Interface web interativa
    - `Plotly`: Visualizações dinâmicas
    
    **Modelo Campeão**: LSTM (R² = ~84%)
    
    O modelo foi treinado com dados dos últimos 90 dias e utiliza janelas de 20 dias para prever o próximo fechamento.
    
    **Fonte de dados**: [CoinGecko API](https://www.coingecko.com/en/api)
    """)

    st.markdown("---")
    st.caption("Desenvolvido por: Alex Soares da Silva, RM354660")