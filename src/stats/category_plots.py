import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import os

def cargar_datos_categorias(level, output_dir="./Output"):
    """Carga y une los CSVs de etiquetas y valores por categoría."""
    df_etiquetas = pd.read_csv(os.path.join(output_dir, f"categorias_{level}_simplificado_etiquetas.csv"))
    df_valores = pd.read_csv(os.path.join(output_dir, f"categorias_{level}_simplificado_valores.csv"))
    return pd.merge(df_valores, df_etiquetas, on='year')

def plot_evolucion_categorias(df_combinado, config_features, seleccion, usar_lowess=False, suavizado=0.2):
    """Grafica la evolución con subplots (unw vs wdocs), soportando suavizado LOWESS."""
    colores_cat = {
        'alta': 'red', 'media_alta': 'orange', 
        'media': 'gold', 'baja': 'royalblue'
    }
    
    feat_cfg = config_features[seleccion]

    # Construcción de diccionarios de argumentos 
    args_scatter_1 = dict(data_frame=df_combinado, x='year', y=feat_cfg['unw_val'], color=feat_cfg['unw_cat'], color_discrete_map=colores_cat)
    args_scatter_2 = dict(data_frame=df_combinado, x='year', y=feat_cfg['wdocs_val'], color=feat_cfg['wdocs_cat'], color_discrete_map=colores_cat)

    if usar_lowess:
        args_scatter_1.update(trendline='lowess', trendline_options=dict(frac=suavizado), trendline_scope='overall')
        args_scatter_2.update(trendline='lowess', trendline_options=dict(frac=suavizado), trendline_scope='overall')

    # Gráfica unw
    fig1 = px.scatter(**args_scatter_1)
    fig1.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # Grafica weighted
    fig2 = px.scatter(**args_scatter_2)
    fig2.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))

    # Aplicación de líneas según la bandera elegida
    if usar_lowess:
        for fig in [fig1, fig2]:
            for trace in fig.data:
                if trace.mode == 'lines':
                    trace.line.color = 'rgba(0, 0, 0, 0.6)'
                    trace.line.width = 3
    else:
        fig1.add_scatter(x=df_combinado['year'], y=df_combinado[feat_cfg['unw_val']], mode='lines', line=dict(color='lightgray', width=1), showlegend=False, hoverinfo='skip')
        fig2.add_scatter(x=df_combinado['year'], y=df_combinado[feat_cfg['wdocs_val']], mode='lines', line=dict(color='lightgray', width=1), showlegend=False, hoverinfo='skip')

    fig_final = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Valores de {feat_cfg["titulo"]}', f'Valores de {feat_cfg["titulo"]} (wdocs)'),
        vertical_spacing=0.1
    )

    for trace in fig1.data:
        fig_final.add_trace(trace, row=1, col=1)

    for trace in fig2.data:
        if trace.name is not None:
            trace.showlegend = False 
        fig_final.add_trace(trace, row=2, col=1)

    fig_final.update_layout(
        height=800, template='plotly_white',
        title_text=f'Evolución de Categorías: {feat_cfg["titulo"]}',
        hovermode='closest'
    )
    fig_final.update_yaxes(title_text="Valor Calculado", row=1, col=1)
    fig_final.update_yaxes(title_text="Valor Calculado (wdocs)", row=2, col=1)

    fig_final.show()