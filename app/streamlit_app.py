# Passos Mágicos: Aplicação de Predição de Risco e Defasagem
# Desenvolvida como entrega do Datathon POSTECH - Fase 5

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib 
import os

# CONFIGURAÇÃO DA PÁGINAS
st.set_page_config(
    page_title="Passos Mágicos - Risco de Defasagem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# PALETA DE CORES
COR_QUARTZO  = '#95A5A6'
COR_AGATA    = '#2ECC71'
COR_AMETISTA = '#9B59B6'
COR_TOPAZIO  = '#3498DB'
COR_RISCO    = '#E74C3C'
COR_OK       = '#27AE60'
COR_ALERTA   = '#F39C12'

# FUNÇÕES UTILITÁRIAS

@st.cache_resource
def carregar_modelo():
    # Carrega modelo, features e threshold do disco
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    modelo     = joblib.load(os.path.join(base, 'modelo_risco.pkl'))
    features   = joblib.load(os.path.join(base, 'features.pkl'))
    threshold  = joblib.load(os.path.join(base, 'threshold_otimo.pkl'))
    return modelo, features, threshold

@st.cache_data
def carregar_dados():
    # Carrega o dataset consolidado
    base = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    df = pd.read_parquet(os.path.join(base, 'pede_consolidado.parquet'))
    ordem_pedras = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    df['PEDRA'] = pd.Categorical(
        df['PEDRA'], categories=ordem_pedras, ordered=True
    )
    return df

def classificar_risco(prob, threshold):
    # Retorna label, cor e emoji baseado na probabilidade
    if prob < 0.30:
        return '🟢 SEM RISCO', COR_OK, 'success'
    elif prob < threshold:
        return '🟡 RISCO INCERTO', COR_ALERTA, 'warning'
    else:
        return '🔴 EM RISCO', COR_RISCO, 'error'
    
def calcular_features(inputs: dict, df_ref: pd.DataFrame) -> pd.DataFrame:
    # Recebe os valores brutos inseridos pelo usuário e calcula todas as features derivadas que o modelo espera
    ida  = inputs['IDA']
    ieg  = inputs['IEG']
    iaa  = inputs['IAA']
    ips  = inputs['IPS']
    ipv  = inputs['IPV']
    ian  = inputs['IAN']
    ipp  = inputs.get('IPP', None)
    fase = inputs['FASE_NUM']
    ano  = inputs['ANO']

    # Média de IDA da fase/ano no dataset histórico (para IDA_vs_MEDIA_FASE)
    subset = df_ref[
        (df_ref['FASE_NUM'] == fase) & (df_ref['ANO'] == ano)
    ]['IDA']
    media_fase = subset.mean() if len(subset) > 0 else df_ref['IDA'].mean()

    row = {
        'IDA':              ida,
        'IEG':              ieg,
        'IAA':              iaa,
        'IPS':              ips,
        'IPV':              ipv,
        'IAN_BIN':          int(ian > 7),
        'GAP_IAA_IDA':      iaa - ida,
        'IAA_SEM_RESP':     int(iaa < 1),
        'IDA_vs_MEDIA_FASE':ida - media_fase,
        'IPP_FILL':         ipp if ipp is not None else df_ref['IPP'].median(),
        'IPP_DISPONIVEL':   int(ipp is not None),
        'FASE_NUM':         fase,
        'ANO':              ano,
        'IDA_x_IEG':        ida * ieg,
        'IAN_x_IDA':        ian * ida,
    }
    return pd.DataFrame([row])

# CARREGAMENTO GLOBAL

modelo, FEATURES, THRESHOLD = carregar_modelo()
df = carregar_dados()

# Calcular FASE_NUM se não existir no parquet
mapa_fase = {
    'ALFA': 0, 'FASE 1': 1, 'FASE 2': 2, 'FASE 3': 3,
    'FASE 4': 4, 'FASE 5': 5, 'FASE 6': 6, 'FASE 7': 7, 'FASE 8': 8
}
if 'FASE_NUM' not in df.columns:
    df['FASE_NUM'] = df['FASE'].map(mapa_fase)

# SIDEBAR - NAVEGAÇÃO

with st.sidebar:
    st.image(
        "https://passosmagicos.org.br/wp-content/uploads/2020/10/Passos-magicos-icon-cor.png",
        width=200
    )
    st.markdown("---")
    st.markdown("### Navegação")
    pagina = st.radio(
        label="Selecione a página:",
        options=["Visão Geral", "Predição Individual", "Dashboard Analítico"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "**Datathon POSTECH - Fase 5**  \n"
        "Associação Passos Mágicos  \n"
        "Dataset: 2022 · 2023 · 2024"
    )
    st.caption(f"Threshold do modelo: {THRESHOLD:.3f}")

# PÁGINA 1: VISÃO GERAL

if pagina == "Visão Geral":

    st.title("Passos Mágicos - Monitoramento de Risco de Defasagem")
    st.markdown(
        "Esta plataforma analisa o desenvolvimento educacional dos alunos da "
        "**Associação Passos Mágicos** utilizando dados do PEDE (2022-2024) e "
        "um modelo preditivo de risco de defasagem baseado em Machine Learning."
    )

    # KPIs principais
    st.markdown("### Indicadores Gerais")
    total = len(df)
    em_risco = df['RISCO'].sum() if 'RISCO' in df.columns else 0
    taxa_risco = em_risco / total * 100 if total > 0 else 0
    topazios  = (df['PEDRA'] == 'Topázio').sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Alunos", f"{total:,.0f}")
    col2.metric("Em Risco de Defasagem", f"{em_risco:,.0f}",
                delta=f"{taxa_risco:.1f}% do total", delta_color="inverse")
    col3.metric("Alunos Topázio", f"{topazios:,.0f}",
                delta=f"{topazios/total*100:.1f}% do total")
    col4.metric("Anos Analisados", "3 (2022–2024)")

    st.markdown("---")

    # Distribuição por Pedra e Ano
    st.markdown("### Distribuição por Pedra e Ano")

    ordem_pedras = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
    cores_pedra  = [COR_QUARTZO, COR_AGATA, COR_AMETISTA, COR_TOPAZIO]
    pedra_ano    = df.groupby(['ANO', 'PEDRA'], observed=True).size().reset_index(name='Qtde')

    anos   = sorted(df['ANO'].unique())
    pedras = [p for p in ordem_pedras if p in df['PEDRA'].cat.categories]
    x      = np.arange(len(pedras))
    width  = 0.25
    cores_anos = ['#4C72B0', '#DD8452', '#55A868']

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ano in enumerate(anos):
        dados_ano = pedra_ano[pedra_ano['ANO'] == ano]
        qtdes = [
            dados_ano[dados_ano['PEDRA'] == p]['Qtde'].values[0]
            if p in dados_ano['PEDRA'].values else 0
            for p in pedras
        ]
        bars = ax.bar(x + i * width, qtdes, width,
                      label=str(ano), color=cores_anos[i])
        ax.bar_label(bars, fontsize=7, padding=2)

    ax.set_xticks(x + width)
    ax.set_xticklabels(pedras)
    ax.set_ylabel('Quantidade de Alunos')
    ax.legend(title='Ano')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Evolução dos indicadores
    st.markdown("### Evolução dos Indicadores (2022–2024)")

    indicadores = ['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'INDE']
    ind_sel = st.multiselect(
        "Selecione os indicadores:",
        options=indicadores,
        default=['IDA', 'IEG', 'IAA', 'INDE']
    )

    if ind_sel:
        evolucao = df.groupby('ANO')[ind_sel].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        cores_ind = plt.cm.tab10.colors
        for i, ind in enumerate(ind_sel):
            ax2.plot(evolucao['ANO'], evolucao[ind], marker='o',
                     label=ind, color=cores_ind[i % 10], linewidth=2)
        ax2.set_xticks([2022, 2023, 2024])
        ax2.set_ylabel('Média')
        ax2.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # Tabela de alunos em risco
    st.markdown("### Alunos com Maior Probabilidade de Risco")
    if 'PROB_RISCO' in df.columns:
        colunas_tabela = ['RA', 'ANO', 'FASE', 'PEDRA', 'IDA', 'IEG',
                          'INDE', 'PROB_RISCO']
        colunas_disp = [c for c in colunas_tabela if c in df.columns]
        top_risco = (
            df[colunas_disp]
            .dropna(subset=['PROB_RISCO'])
            .sort_values('PROB_RISCO', ascending=False)
            .head(20)
        )
        top_risco['PROB_RISCO'] = top_risco['PROB_RISCO'].round(3)
        st.dataframe(top_risco, use_container_width=True)
    else:
        st.info("Execute o notebook 03_model.ipynb para gerar as probabilidades de risco.")

# PÁGINA 2: PREDIÇÃO INDIVIDUAL

elif pagina == "Predição Individual":

    st.title("Predição de Risco - Aluno Individual")
    st.markdown(
        "Insira os indicadores de um aluno para calcular a **probabilidade de risco "
        "de defasagem** em tempo real. O modelo foi treinado com dados de 2022 a 2024."
    )

    # Formulário de entrada
    with st.form("form_predicao"):
        st.markdown("### Dados do Aluno")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Contexto**")
            ano_input  = st.selectbox("Ano de referência", [2022, 2023, 2024], index=2)
            fase_input = st.selectbox(
                "Fase",
                options=list(mapa_fase.keys()),
                index=3
            )
            ipp_input = st.slider(
                "IPP (Psicopedagógico) - deixe 0 se não disponível",
                min_value=0.0, max_value=10.0, value=0.0, step=0.1
            )
            ipp_disponivel = ipp_input > 0

        with col2:
            st.markdown("**Indicadores de Desempenho**")
            ida_input = st.slider("IDA (Desempenho Acadêmico)",
                                  0.0, 10.0, 6.0, 0.1)
            ieg_input = st.slider("IEG (Engajamento)",
                                  0.0, 10.0, 7.0, 0.1)
            ian_input = st.slider("IAN (Adequação ao Nível)",
                                  0.0, 10.0, 7.0, 0.5)
            ipv_input = st.slider("IPV (Ponto de Virada)",
                                  0.0, 10.0, 7.0, 0.1)

        with col3:
            st.markdown("**Indicadores Comportamentais**")
            iaa_input = st.slider("IAA (Autoavaliação)",
                                  0.0, 10.0, 7.0, 0.1)
            ips_input = st.slider("IPS (Psicossocial)",
                                  0.0, 10.0, 6.25, 0.25)
            
        submitted = st.form_submit_button(
            "Calcular Probabilidade de Risco",
            use_container_width=True,
            type="primary"
        )

    # Resultado da predição
    if submitted:
        inputs = {
            'IDA':      ida_input,
            'IEG':      ieg_input,
            'IAA':      iaa_input,
            'IPS':      ips_input,
            'IPV':      ipv_input,
            'IAN':      ian_input,
            'IPP':      ipp_input if ipp_disponivel else None,
            'FASE_NUM': mapa_fase[fase_input],
            'ANO':      ano_input,
        }

        X_pred = calcular_features(inputs, df)
        prob   = modelo.predict_proba(X_pred[FEATURES])[0][1]
        label, cor, tipo = classificar_risco(prob, THRESHOLD)

        st.markdown("---")
        st.markdown("### Resultado")

        col_res1, col_res2, col_res3 = st.columns([1, 1, 2])
        
        with col_res1:
            st.metric(
                label="Probabilidade de Risco",
                value=f"{prob*100:.1f}%"
            )

        with col_res2:
            st.metric(
                label="Classificação",
                value=label
            )

        with col_res3:
            # Gauge visual simples
            fig_g, ax_g = plt.subplots(figsize=(5, 1.5))
            ax_g.barh([''], [prob], color=cor, height=0.4)
            ax_g.barh([''], [1 - prob], left=[prob],
                      color='#ECF0F1', height=0.4)
            ax_g.axvline(THRESHOLD, color='navy', linestyle='--',
                         linewidth=1.5, label=f'Threshold ({THRESHOLD:.2f})')
            ax_g.set_xlim(0, 1)
            ax_g.set_xlabel('P(Risco)')
            ax_g.legend(fontsize=8)
            ax_g.set_title(f'P(Risco) = {prob:.3f}', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_g)
            plt.close()

        # Mensagem contextual
        if tipo == 'error':
            st.error(
                f"**{label}** - P(Risco) = {prob*100:.1f}%  \n"
                "Este aluno apresenta indicadores consistentes com risco de defasagem. "
                "Recomenda-se intervenção pedagógica e psicopedagógica imediata."
            )
        elif tipo == 'warning':
            st.warning(
                f"**{label}** - P(Risco) = {prob*100:.1f}%  \n"
                "Este aluno está na zona de atenção. "
                "Monitore de perto os próximos ciclos de avaliação."
            )
        else:
            st.success(
                f"**{label}** - P(Risco) = {prob*100:.1f}%  \n"
                "Os indicadores deste aluno estão dentro de um padrão saudável de desenvolvimento."
            )

        st.markdown("---")

        # Radar dos indicadores do aluno
        st.markdown("#### Perfil do Aluno vs Média da Base")

        cats  = ['IDA', 'IEG', 'IAA', 'IPS', 'IPV']
        vals_aluno = [ida_input, ieg_input, iaa_input, ips_input, ipv_input]
        vals_media = [df[c].mean() for c in cats]

        angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
        angles += angles[:1]
        v_aluno = vals_aluno + vals_aluno[:1]
        v_media = vals_media + vals_media[:1]

        fig_r, ax_r = plt.subplots(figsize=(5, 5),
                                   subplot_kw=dict(polar=True))
        ax_r.plot(angles, v_aluno, 'o-', linewidth=2,
                  color=cor, label='Aluno')
        ax_r.fill(angles, v_aluno, alpha=0.2, color=cor)
        ax_r.plot(angles, v_media, 'o-', linewidth=2,
                  color='#7F8C8D', linestyle='--', label='Média base')
        ax_r.fill(angles, v_media, alpha=0.05, color='#7F8C8D')
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(cats, fontsize=10)
        ax_r.set_ylim(0, 10)
        ax_r.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax_r.set_title('Perfil do Aluno', fontsize=11, pad=15)
        plt.tight_layout()
        st.pyplot(fig_r)
        plt.close()

        # ── Tabela de features calculadas ───────────────────
        with st.expander("Ver features calculadas para o modelo"):
            st.dataframe(X_pred[FEATURES].T.rename(columns={0: 'Valor'}),
                         use_container_width=True)
    
# PÁGINA 3: DASHBOARD ANALÍTICO

elif pagina == "Dashboard Analítico":
    
    st.title("Dashboard Analítico - Passos Mágicos")
    st.markdown("Análise dos indicadores por ano, fase e classificação (Pedra).")

    # Filtros
    st.markdown("### Filtros")
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        anos_sel = st.multiselect(
            "Ano", options=sorted(df['ANO'].unique()),
            default=sorted(df['ANO'].unique())
        )
    with col_f2:
        pedras_disp = [p for p in ['Quartzo', 'Ágata', 'Ametista', 'Topázio']
                       if p in df['PEDRA'].cat.categories]
        pedras_sel = st.multiselect(
            "Pedra", options=pedras_disp, default=pedras_disp
        )
    with col_f3:
        indicador_sel = st.selectbox(
            "Indicador principal",
            options=['IDA', 'IEG', 'IAA', 'IPS', 'IPV', 'IPP', 'IAN', 'INDE'],
            index=0
        )
    
    df_filt = df[
        df['ANO'].isin(anos_sel) &
        df['PEDRA'].isin(pedras_sel)
    ].copy()

    if df_filt.empty:
        st.warning("Nenhum dado para os filtros selecionados.")
        st.stop()

    st.markdown(f"*{len(df_filt):,} registros selecionados*")
    st.markdown("---")

    # Linha 1: Boxplot por Pedra + evolução temporal
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"#### Distribuição de {indicador_sel} por Pedra")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        cores_p = [COR_QUARTZO, COR_AGATA, COR_AMETISTA, COR_TOPAZIO]
        pedras_plot = [p for p in pedras_disp if p in pedras_sel]
        sns.boxplot(
            data=df_filt, x='PEDRA', y=indicador_sel,
            order=pedras_plot,
            palette=dict(zip(pedras_plot, cores_p[:len(pedras_plot)])),
            ax=ax1
        )
        ax1.set_xlabel('Pedra')
        ax1.set_ylabel(indicador_sel)
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    with col2:
        st.markdown(f"#### Evolução de {indicador_sel} por Pedra")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        cores_map = dict(zip(pedras_disp, [COR_QUARTZO, COR_AGATA,
                                            COR_AMETISTA, COR_TOPAZIO]))
        evol = df_filt.groupby(['ANO', 'PEDRA'], observed=True)[indicador_sel].mean().reset_index()
        for pedra in pedras_sel:
            sub = evol[evol['PEDRA'] == pedra]
            if len(sub) > 0:
                ax2.plot(sub['ANO'], sub[indicador_sel], marker='o',
                         label=pedra, color=cores_map.get(pedra, 'gray'),
                         linewidth=2)
        ax2.set_xticks(anos_sel if len(anos_sel) > 1 else [2022, 2023, 2024])
        ax2.set_ylabel(indicador_sel)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # Linha 2: Correlação + risco por Pedra
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Correlação entre Indicadores")
        indicadores_corr = ['IDA', 'IEG', 'IAA', 'IPS', 'IPV', 'IAN', 'INDE']
        corr = df_filt[indicadores_corr].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', vmin=-1, vmax=1,
                    linewidths=0.5, ax=ax3, annot_kws={'size': 8})
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col4:
        st.markdown("#### Taxa de Risco por Pedra e Ano")
        if 'RISCO' in df_filt.columns:
            risco_pedra = (
                df_filt.groupby(['ANO', 'PEDRA'], observed=True)['RISCO']
                .mean()
                .reset_index()
            )
            risco_pedra['Taxa (%)'] = risco_pedra['RISCO'] * 100

            fig4, ax4 = plt.subplots(figsize=(6, 4))
            for pedra in pedras_sel:
                sub = risco_pedra[risco_pedra['PEDRA'] == pedra]
                if len(sub) > 0:
                    ax4.plot(sub['ANO'], sub['Taxa (%)'], marker='o',
                             label=pedra, color=cores_map.get(pedra, 'gray'),
                             linewidth=2)
            ax4.set_xticks([2022, 2023, 2024])
            ax4.set_ylabel('Taxa de Risco (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig4)
            plt.close()
        else:
            st.info("Execute o notebook 03_model.ipynb para calcular o risco.")

    st.markdown("---")

    # Linha 3: Estatísticas descritivas
    st.markdown("#### Estatísticas Descritivas por Pedra")
    indicadores_desc = ['IDA', 'IEG', 'IAA', 'IPS', 'IPV', 'INDE']
    stats = (
        df_filt.groupby('PEDRA', observed=True)[indicadores_desc]
        .mean()
        .round(2)
    )
    st.dataframe(stats, use_container_width=True)
