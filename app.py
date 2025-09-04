import streamlit as st # Biblioteca principal do Streamlit
import pandas as pd # Biblioteca para manipulação de dados
from datetime import datetime
import time
from web_scraper import buscar_dados_brasileirao # Importa os dados do web scraper
from feature_engineering import preparar_dados_para_modelo # Importa a preparação dos dados
from model_trainer import treinar_modelo # Importa o treinamento do modelo
from predictor import prever_jogo_especifico, simular_campeonato # Importa as funções de previsão e simulação
from analysis import gerar_confronto_direto # Importa a análise de confronto direto

# Página
st.set_page_config(
    page_title="FUTBot | Previsão de Futebol",
    page_icon="⚽",
    layout="wide"
)

@st.cache_data
def carregar_dados_e_modelo():
    """
    Função para BAIXAR dados (APENAS DE 2025), preparar features e treinar o modelo.
    """
    with st.spinner('Baixando dados da temporada 2025... Isso pode levar um momento.'):
        ano_atual = datetime.now().year
        df_total = buscar_dados_brasileirao(str(ano_atual))
    
    if df_total is None:
        st.error("Falha ao buscar os dados de 2025. Tente recarregar a página.")
        return (None,) * 8

    df_resultados = df_total[df_total['FTHG'].notna()].copy()
    df_futuro = df_total[df_total['FTHG'].isna()].copy()

    with st.spinner('Calculando features e treinando o modelo com dados de 2025...'):
        df_treino, time_stats = preparar_dados_para_modelo(df_resultados)
        if df_treino.empty:
            st.warning("Ainda não há dados de treino suficientes na temporada para treinar um modelo.")
            return (None,) * 8
        modelo, encoder, colunas_modelo = treinar_modelo(df_treino)

    lista_times = sorted(list(set(df_total['HomeTeam']).union(set(df_total['AwayTeam']))))
    
    st.success("Tudo pronto! Modelo treinado e dados de 2025 carregados.")
    return df_resultados, df_futuro, time_stats, modelo, encoder, colunas_modelo, lista_times, df_total

# Carrega os dados cache
(df_resultados, df_futuro, time_stats, 
 modelo, encoder, colunas_modelo, 
 lista_times, df_total) = carregar_dados_e_modelo()

# Interface do Usuário 
st.title("FUTBot: Previsões do Brasileirão 2025")

st.sidebar.header("Menu de Opções")
menu_escolha = st.sidebar.radio(
    "O que você gostaria de fazer?",
    ("Prever o resultado de um jogo", "Simular a classificação do campeonato", "Confronto Direto")
)

if df_resultados is not None and modelo is not None:

    # Previsão de Jogo Específico
    if menu_escolha == "Prever o resultado de um jogo":
        st.header("Previsão de Jogo Específico")
        st.markdown("Selecione dois times para ver as probabilidades de vitória, empate ou derrota para a partida.")
        
        col1, col2 = st.columns(2)
        with col1:
            time_casa = st.selectbox("Escolha o time da casa:", lista_times or [], index=None, placeholder="Selecione o time...")
        with col2:
            time_visitante = st.selectbox("Escolha o time visitante:", lista_times or [], index=None, placeholder="Selecione o time...")
            
        if st.button("Prever Resultado", use_container_width=True, type="primary"): # Botão para prever o jogo
            if time_casa and time_visitante:
                if time_casa == time_visitante: st.warning("O time da casa e o visitante devem ser diferentes.")
                else:
                    with st.spinner('Calculando probabilidades...'):
                        odds = prever_jogo_especifico(time_casa, time_visitante, modelo, encoder, time_stats, colunas_modelo)
                    st.subheader(f"Previsão para: {time_casa} vs {time_visitante}")
                    resultado_provavel = max(odds, key=lambda k: odds[k])
                    for resultado, prob in odds.items():
                        if prob > 0: odd_decimal = 1 / prob; texto_odd = f"({odd_decimal:.2f})"
                        else: texto_odd = ""
                        if resultado == resultado_provavel: st.markdown(f"**🏆 {resultado}: {prob:.1%} {texto_odd}**")
                        else: st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{resultado}: {prob:.1%} {texto_odd}")
            else: st.error("Por favor, selecione os dois times para fazer a previsão.")

    # Simulação do Campeonato
    elif menu_escolha == "Simular a classificação do campeonato":
        st.header("Simulação da Tabela do Campeonato")
        st.markdown("Simule os resultados de todos os jogos futuros até uma rodada específica e veja a provável classificação final do campeonato.")
        
        rodada_atual = 0
        if not df_resultados.empty: rodada_atual = int(pd.to_numeric(df_resultados['Rodada']).max())
        rodada_simulacao = st.slider(f"Simular até qual rodada? (Rodada atual: {rodada_atual})", min_value=rodada_atual if rodada_atual > 0 else 1, max_value=38, value=38)
        
        if st.button("Simular Tabela", use_container_width=True, type="primary"):
            with st.spinner(f"Simulando todos os jogos até a rodada {rodada_simulacao}..."):
                tabela_simulada = simular_campeonato(rodada_simulacao, df_futuro, df_resultados, modelo, encoder, time_stats, colunas_modelo)
            st.success(f"Tabela de classificação simulada até a rodada {rodada_simulacao}:")
            st.dataframe(tabela_simulada, hide_index=True, use_container_width=True)
    
    # Confronto Direto        
    elif menu_escolha == "Confronto Direto":
        st.header("Confronto Direto")
        st.markdown("Analise o histórico completo de partidas entre dois clubes, incluindo vitórias, empates e gols.")
        
        col1, col2 = st.columns(2)
        with col1:
            time1 = st.selectbox("Escolha o primeiro time:", lista_times or [], index=None, key="time1_h2h", placeholder="Selecione o time...")
        with col2:
            time2 = st.selectbox("Escolha o segundo time:", lista_times or [], index=None, key="time2_h2h", placeholder="Selecione o time...")

        # Botão para analisar o confronto
        if st.button("Analisar Confronto", use_container_width=True, type="primary"):
            if time1 and time2:
                if time1 == time2:
                    st.warning("Por favor, escolha dois times diferentes.")
                else:
                    with st.spinner("Buscando histórico de confrontos..."):
                        resumo, historico_df = gerar_confronto_direto(df_total, time1, time2)
                    
                    if resumo is None:
                        st.info(f"Não foram encontrados jogos entre {time1} e {time2} nos dados carregados.")
                    else:
                        st.subheader(f"Resumo do Confronto: {time1} vs {time2}")
              
                        # Exibe as métricas principais em colunas
                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        col_res1.metric(f"Vitórias {time1}", resumo['vitorias'][time1])
                        col_res2.metric("Empates", resumo['empates'])
                        col_res3.metric(f"Vitórias {time2}", resumo['vitorias'][time2])
                        col_res4.metric("Total de Partidas", resumo['total_partidas']) # <-- Nova Métrica

                        col_gols1, col_gols2 = st.columns(2)
                        col_gols1.metric(f"Gols Marcados por {time1} (2025)", resumo['gols'][time1])
                        col_gols2.metric(f"Gols Marcados por {time2} (2025)", resumo['gols'][time2])

                        st.subheader("Histórico de Jogos Recentes")
                        st.dataframe(historico_df, hide_index=True, use_container_width=True)
            else:
                st.error("Por favor, selecione os dois times para fazer a análise.")
else:
    st.error("Não foi possível carregar os dados ou treinar o modelo.")