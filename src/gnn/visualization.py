import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd

def plot_trayectoria_2d(df_trayectoria):
    """Grafica la trayectoria 2D de la evolución de los grafos."""
    fig = px.scatter(df_trayectoria, x='x', y='y', 
                     text='label', 
                     color='year',
                     title="Evolución PCA 2D")
    fig.update_traces(mode='lines+markers+text', textposition='top center')
    fig.show()

def plot_evolucion_pc1(df_flujo):
    """Grafica la línea de tiempo del Componente Principal 1."""
    fig = px.line(
        df_flujo, 
        x='year', 
        y='y_original',
        title="Evolución PC1 de GraphSAGE",
        labels={'y_original': 'PC1', 'year': 'Año'},
        markers=True,
        template="plotly_white"
    )
    fig.show()

def plot_correlaciones_barras(correlaciones):
    """Muestra un gráfico de barras horizontales de las correlaciones de Pearson."""
    plt.figure(figsize=(10, 6))
    sorted_corr = pd.Series(correlaciones).sort_values()#para que el valor mas alto aparezca arriba
    colors = ['red' if x < 0 else 'blue' for x in sorted_corr]
    sorted_corr.plot(kind='barh', color=colors)
    plt.title("Correlación entre Features y Trayectoria (PC1)")
    plt.xlabel("Correlación de Pearson")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.show()

def plot_feature_individual(df_plot, atributo_a_graficar="Percent of documents"):
    """Grafica la evolución de una feature específica a partir de los datos crudos."""
    fig = px.line(
        df_plot, 
        x='Year', 
        y=atributo_a_graficar,
        title=f"Evolución de: {atributo_a_graficar}",
        markers=True,
        template="plotly_white"
    )
    fig.show()

# Normalización Manual (Escala 0 a 1)
# Para poder comparar directamente las tendencias
#PD es percent of documents

#Por cada atributo definido, se busca al maximo, y a cada valor (fila) es dividido entre ese valor
def plot_comparacion_normalizada(df_comparacion):
    """Grafica la comparativa de intensidad relativa (0 a 1)."""
    fig = px.line(
        df_comparacion, 
        x='Year', 
        y=['PD Crudos', 'PD PCA', 'Autores'], 
        title="Comparación de evoluciones de los embeddings contra los datos crudos",
        labels={'value': 'Intensidad Relativa (0-1)', 'variable': 'Curva', 'Year': 'Año'},
        template="plotly_white",
        markers=False
    )
    fig.update_traces(line=dict(width=3)) 
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector={"name": "Autores"})
    fig.update_traces(patch={"line": {"dash": "dot"}}, selector={"name": "PD Crudos"})
    fig.show()