import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

#Pendiente de unificar funciones redundantes
def cargar_datos_categorias(level, output_dir="./Output"):
    """Carga y une los CSVs de etiquetas y valores por categoría."""
    df_etiquetas = pd.read_csv(os.path.join(output_dir, f"categorias_{level}_simplificado_etiquetas.csv"))
    df_valores = pd.read_csv(os.path.join(output_dir, f"categorias_{level}_simplificado_valores.csv"))
    return pd.merge(df_valores, df_etiquetas, on='year')

def cargar_datos_completos(level, output_dir="./Output"):
    """Carga y une los CSVs de etiquetas, valores Y proporciones detalladas."""
    df_etiq = pd.read_csv(os.path.join(output_dir, f"categorias_{level}_simplificado_etiquetas.csv"))
    df_val = pd.read_csv(os.path.join(output_dir, f"categorias_{level}_simplificado_valores.csv"))
    df_det = pd.read_csv(os.path.join(output_dir, f"categorias_{level}_detallado.csv"))
    
    df_merged = pd.merge(df_val, df_etiq, on='year')
    df_completo = pd.merge(df_merged, df_det, on='year')
    return df_completo

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


def plot_evolucion_completa(df_completo, config_features, seleccion):
    """
    Grafica la evolución añadiendo, detrás de los puntos/líneas,
    las barras agrupadas (barmode='group') con todas las proporciones
    de cada categoría en ese año.
    """
    colores_cat = {
        'alta': 'red', 'media_alta': 'orange', 
        'media': 'gold', 'baja': 'royalblue'
    }
    todas_etiquetas = ['baja', 'media', 'media_alta', 'alta']
    
    feat_cfg = config_features[seleccion]
    base_unw = feat_cfg['unw_cat']
    base_wdocs = feat_cfg['wdocs_cat']

    fig_final = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Valores Proporcionales de {feat_cfg["titulo"]}', f'Valores de {feat_cfg["titulo"]} (wdocs)'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Añadir BARRAS (todas las categorías)
    for etiq in todas_etiquetas:
        col_unw = f"{base_unw}_{etiq}"
        col_wdocs = f"{base_wdocs}_{etiq}"
        color = colores_cat[etiq]
        
        # Subplot 1 (unw)
        if col_unw in df_completo.columns:
            fig_final.add_trace(
                go.Bar(x=df_completo['year'], y=df_completo[col_unw], 
                       name=etiq, marker_color=color, opacity=0.6,
                       legendgroup=etiq, showlegend=True),
                row=1, col=1
            )
            
        # Subplot 2 (wdocs)
        if col_wdocs in df_completo.columns:
            fig_final.add_trace(
                go.Bar(x=df_completo['year'], y=df_completo[col_wdocs], 
                       name=etiq, marker_color=color, opacity=0.6,
                       legendgroup=etiq, showlegend=False),
                row=2, col=1
            )

    # Añadir LÍNEAS Y PUNTOS del Dominante (scatter)
    
    # Crear scatter para unw
    scatter_unw = px.scatter(df_completo, x='year', y=feat_cfg['unw_val'], 
                             color=feat_cfg['unw_cat'], color_discrete_map=colores_cat)
    scatter_unw.update_traces(marker=dict(size=12, line=dict(width=2, color='black')))
    
    # Añadir línea gris que une los puntos (unw)
    fig_final.add_trace(
        go.Scatter(x=df_completo['year'], y=df_completo[feat_cfg['unw_val']],
                   mode='lines', line=dict(color='gray', width=1.5),
                   showlegend=False, hoverinfo='skip'),
        row=1, col=1
    )
    
    # Añadir los puntos del scatter (unw)
    for trace in scatter_unw.data:
        trace.showlegend = False # Ocultar leyenda para no duplicar con las barras
        fig_final.add_trace(trace, row=1, col=1)

    # Crear scatter para wdocs
    scatter_w = px.scatter(df_completo, x='year', y=feat_cfg['wdocs_val'], 
                           color=feat_cfg['wdocs_cat'], color_discrete_map=colores_cat)
    scatter_w.update_traces(marker=dict(size=12, line=dict(width=2, color='black')))

    # Añadir línea gris que une los puntos (wdocs)
    fig_final.add_trace(
        go.Scatter(x=df_completo['year'], y=df_completo[feat_cfg['wdocs_val']],
                   mode='lines', line=dict(color='gray', width=1.5),
                   showlegend=False, hoverinfo='skip'),
        row=2, col=1
    )
    
    # Añadir los puntos del scatter (wdocs)
    for trace in scatter_w.data:
        trace.showlegend = False 
        fig_final.add_trace(trace, row=2, col=1)

    # Configuración del layout final
    fig_final.update_layout(
        height=900, 
        template='plotly_white',
        title_text=f'Evolución Completa: {feat_cfg["titulo"]} (Distribución Total vs Ganador)',
        hovermode='x unified',
        barmode='group'  # Agrupa las barras anualmente para que no se superpongan
    )
    
    fig_final.update_yaxes(title_text="Proporción (0-1)", row=1, col=1)
    fig_final.update_yaxes(title_text="Proporción wdocs (0-1)", row=2, col=1)

    fig_final.show()

def plot_peak_dominance(df_completo, config_features, seleccion):
    """
    Grafica la evolución con diseño 'Light & Vibrant' para publicaciones científicas.
    Usa barras apiladas como contexto de fondo y una línea oscura flotante 
    para destacar la cuota de la categoría ganadora.
    """
    # PALETA VIBRANTE
    colores_cat = {
        'baja': '#007BFF',        # Azul eléctrico / brillante
        'media': '#FFC107',       # Amarillo vivo / Oro
        'media_alta': '#FD7E14',  # Naranja intenso
        'alta': '#DC3545'         # Rojo carmesí / vibrante
    }
    todas_etiquetas = ['baja', 'media', 'media_alta', 'alta']
    
    feat_cfg = config_features[seleccion]
    base_unw = feat_cfg['unw_cat']
    base_wdocs = feat_cfg['wdocs_cat']

    fig_final = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{feat_cfg["titulo"]} - Peak Dominance', 
                        f'{feat_cfg["titulo"]} (wdocs) - Peak Dominance'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Añadir BARRAS APILADAS (Fondo contextual)
    for etiq in todas_etiquetas:
        col_unw = f"{base_unw}_{etiq}"
        col_wdocs = f"{base_wdocs}_{etiq}"
        color = colores_cat.get(etiq, 'gray')
        
        # Reducimos un poco la opacidad (0.5 o 0.6) para que los colores vivos 
        # no se "coman" a los puntos principales que irán encima.
        if col_unw in df_completo.columns:
            fig_final.add_trace(
                go.Bar(x=df_completo['year'], y=df_completo[col_unw], 
                       name=etiq.replace('_', ' ').title(), marker_color=color, 
                       marker_line_width=0.5, marker_line_color='white', # Bordes blancos separan mejor
                       opacity=0.55, legendgroup=etiq, showlegend=True),
                row=1, col=1
            )
            
        if col_wdocs in df_completo.columns:
            fig_final.add_trace(
                go.Bar(x=df_completo['year'], y=df_completo[col_wdocs], 
                       name=etiq.replace('_', ' ').title(), marker_color=color, 
                       marker_line_width=0.5, marker_line_color='white',
                       opacity=0.55, legendgroup=etiq, showlegend=False),
                row=2, col=1
            )

    # Añadir LÍNEAS Y PUNTOS (Primer plano)
    
    # Línea oscura para contraste sobre el fondo claro
    color_linea_principal = '#333333' # Gris carbón oscuro
    
    # unw
    fig_final.add_trace(
        go.Scatter(x=df_completo['year'], y=df_completo[feat_cfg['unw_val']],
                   mode='lines', line=dict(color=color_linea_principal, width=2.5),
                   name='PEAK DOMINANCE', legendgroup='peak', showlegend=True, hoverinfo='skip'),
        row=1, col=1
    )
    scatter_unw = px.scatter(df_completo, x='year', y=feat_cfg['unw_val'], 
                             color=feat_cfg['unw_cat'], color_discrete_map=colores_cat)
    # Puntos con colores sólidos al 100% (sin transparencia) para que resalten sobre las barras
    scatter_unw.update_traces(marker=dict(size=12, opacity=1.0, line=dict(width=2, color=color_linea_principal)))
    for trace in scatter_unw.data:
        trace.showlegend = False 
        fig_final.add_trace(trace, row=1, col=1)

    # wdocs
    fig_final.add_trace(
        go.Scatter(x=df_completo['year'], y=df_completo[feat_cfg['wdocs_val']],
                   mode='lines', line=dict(color=color_linea_principal, width=2.5),
                   name='PEAK DOMINANCE', legendgroup='peak', showlegend=False, hoverinfo='skip'),
        row=2, col=1
    )
    scatter_w = px.scatter(df_completo, x='year', y=feat_cfg['wdocs_val'], 
                           color=feat_cfg['wdocs_cat'], color_discrete_map=colores_cat)
    scatter_w.update_traces(marker=dict(size=12, opacity=1.0, line=dict(width=2, color=color_linea_principal)))
    for trace in scatter_w.data:
        trace.showlegend = False 
        fig_final.add_trace(trace, row=2, col=1)

    fig_final.update_layout(
        height=900, 
        template='plotly_white',          # Tema blanco limpio
        plot_bgcolor="white",             # Fondo de la gráfica blanco
        paper_bgcolor="white",            # Margen blanco
        hovermode='x unified',
        barmode='stack',
        font=dict(color="#333333"),       # Letras oscuras
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Cuadrícula muy sutil para no ensuciar la visualización
    color_grid = '#EAEAEA'
    
    fig_final.update_yaxes(title_text="PROPORTION OF MARKET", tickformat=".0%", range=[0, 1], 
                           row=1, col=1, gridcolor=color_grid, zerolinecolor=color_grid)
    fig_final.update_yaxes(title_text="PROPORTION OF MARKET (wdocs)", tickformat=".0%", range=[0, 1], 
                           row=2, col=1, gridcolor=color_grid, zerolinecolor=color_grid)
    fig_final.update_xaxes(gridcolor=color_grid, zerolinecolor=color_grid)
    
    return fig_final


def plot_evolucion_traslapada(df_completo, config_features, seleccion):
    """
    Grafica la evolución usando barras TRASLAPADAS (overlay) ordenadas dinámicamente.
    Para cada año, la categoría más grande se dibuja en el fondo y las más pequeñas 
    al frente. Esto permite ver los 4 valores reales partiendo desde 0.
    """
    colores_cat = {
        'baja': '#007BFF',        # Azul eléctrico
        'media': '#FFC107',       # Amarillo
        'media_alta': '#FD7E14',  # Naranja
        'alta': '#DC3545'         # Rojo
    }
    todas_etiquetas = ['baja', 'media', 'media_alta', 'alta']
    
    feat_cfg = config_features[seleccion]
    base_unw = feat_cfg['unw_cat']
    base_wdocs = feat_cfg['wdocs_cat']

    fig_final = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'Distribución Traslapada - {feat_cfg["titulo"]}', 
                        f'Distribución Traslapada - {feat_cfg["titulo"]} (wdocs)'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Función auxiliar para generar las 4 capas de barras ordenadas por año
    def preparar_rangos(prefix):
        columnas = [f"{prefix}_{etiq}" for etiq in todas_etiquetas]
        cols_existentes = [c for c in columnas if c in df_completo.columns]
        
        rangos_y = {1: [], 2: [], 3: [], 4: []}
        rangos_c = {1: [], 2: [], 3: [], 4: []}
        
        for idx, row in df_completo.iterrows():
            valores = []
            for col in cols_existentes:
                etiq = col.replace(f"{prefix}_", "")
                val = row[col]
                if pd.notna(val):
                    valores.append((val, colores_cat.get(etiq, 'gray')))
            
            # Ordenar de mayor a menor (el mayor va al fondo = Rank 1)
            valores.sort(key=lambda x: x[0], reverse=True)
            
            # Rellenar con ceros invisibles si faltan etiquetas (e.g. autores tiene 3 bins)
            while len(valores) < 4:
                valores.append((0, 'rgba(0,0,0,0)'))
                
            for i in range(4):
                rangos_y[i+1].append(valores[i][0])
                rangos_c[i+1].append(valores[i][1])
                
        return rangos_y, rangos_c

    # AÑADIR BARRAS TRASLAPADAS
    
    ry_unw, rc_unw = preparar_rangos(base_unw)
    ry_wdocs, rc_wdocs = preparar_rangos(base_wdocs)
    
    # Dibujar de capa 1 (fondo) a capa 4 (frente)
    for rank in [1, 2, 3, 4]:
        fig_final.add_trace(
            go.Bar(x=df_completo['year'], y=ry_unw[rank], 
                   marker_color=rc_unw[rank], 
                   marker_line_width=0.5, marker_line_color='white',
                   opacity=0.85, showlegend=False, hoverinfo='x+y'),
            row=1, col=1
        )
        fig_final.add_trace(
            go.Bar(x=df_completo['year'], y=ry_wdocs[rank], 
                   marker_color=rc_wdocs[rank], 
                   marker_line_width=0.5, marker_line_color='white',
                   opacity=0.85, showlegend=False, hoverinfo='x+y'),
            row=2, col=1
        )

    # LEYENDA FICTICIA (Ya que los colores cambian dinámicamente)
    for etiq, col in colores_cat.items():
        if f"{base_unw}_{etiq}" in df_completo.columns:
            fig_final.add_trace(
                go.Bar(x=[None], y=[None], name=etiq.replace('_', ' ').title(), marker_color=col),
                row=1, col=1
            )

    # AÑADIR LÍNEAS PEAK DOMINANCE Y PUNTOS
    color_linea_principal = '#333333'
    
    # unw (ry_unw[1] es exactamente la altura ganadora en ese año)
    fig_final.add_trace(
        go.Scatter(x=df_completo['year'], y=ry_unw[1], 
                   mode='lines+markers', line=dict(color=color_linea_principal, width=2.5),
                   marker=dict(size=12, color=rc_unw[1], line=dict(width=2, color=color_linea_principal)),
                   name='Peak Dominance', showlegend=False, hoverinfo='skip'),
        row=1, col=1
    )
    
    # wdocs
    fig_final.add_trace(
        go.Scatter(x=df_completo['year'], y=ry_wdocs[1], 
                   mode='lines+markers', line=dict(color=color_linea_principal, width=2.5),
                   marker=dict(size=12, color=rc_wdocs[1], line=dict(width=2, color=color_linea_principal)),
                   name='Peak Dominance', showlegend=False, hoverinfo='skip'),
        row=2, col=1
    )

    fig_final.update_layout(
        height=900, 
        template='plotly_white',
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode='x unified',
        barmode='overlay', # Traslapados en lugar de apilados
        font=dict(color="#333333"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    color_grid = '#EAEAEA'
    fig_final.update_yaxes(title_text="PROPORCIÓN DEL MERCADO", range=[0, 1.05], tickformat=".0%", 
                           row=1, col=1, gridcolor=color_grid, zerolinecolor=color_grid)
    fig_final.update_yaxes(title_text="PROPORCIÓN DEL MERCADO (wdocs)", range=[0, 1.05], tickformat=".0%", 
                           row=2, col=1, gridcolor=color_grid, zerolinecolor=color_grid)
    fig_final.update_xaxes(gridcolor=color_grid, zerolinecolor=color_grid)

    return fig_final
    