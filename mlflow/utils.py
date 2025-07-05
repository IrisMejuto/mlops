# ==========================================
# IMPORTS
# ==========================================

# Python standard library
import os
import json
import gzip
import random
import re
import unicodedata
import tarfile
import rarfile
import zipfile

# Data manipulation and analysis
import pandas as pd
import numpy as np
from collections import Counter

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# NLP libraries
from nltk import ngrams
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Machine Learning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.pipeline import Pipeline

# Specialized libraries
from stop_words import get_stop_words
from wordcloud import WordCloud, STOPWORDS
from gensim.models import Word2Vec
import multiprocessing
from typing import List, Optional, Union
import string
import time
import subprocess
import tempfile


# ==========================================
# CONFIGURACIÓN GLOBAL
# ==========================================

RANDOM_SEED = 42

def set_random_seed(seed=RANDOM_SEED):
    """Establece semilla para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"Semilla establecida: {seed}")

# Establecer semilla por defecto
set_random_seed()

# ==========================================
# FUNCIONES DE CARGA DE DATOS
# ==========================================

def load_amazon_reviews_from_zip(zip_path, filename_inside_zip, sample_size=None):
    """
    Carga reviews de Amazon desde un archivo .zip que contiene un archivo .json.gz
    
    Parameters:
    -----------
    zip_path : str
        Ruta al archivo .zip
    filename_inside_zip : str
        Nombre del archivo .json.gz dentro del .zip
    sample_size : int, optional
        Número de reviews a cargar (para evitar problemas de memoria)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con las reviews
    """
    reviews = []
    
    # Abrir el archivo ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extraer el archivo .json.gz dentro del ZIP
        with zip_ref.open(filename_inside_zip) as file:
            # Descomprimir el archivo .gz
            with gzip.open(file, 'rt', encoding='utf-8') as gz_file:
                for i, line in enumerate(gz_file):
                    if sample_size and i >= sample_size:
                        break
                    try:
                        review = json.loads(line)
                        reviews.append(review)
                    except json.JSONDecodeError:
                        continue

    print(f"Dataset cargado exitosamente: {len(reviews)} reviews")
    return pd.DataFrame(reviews)

# ==========================================
# FUNCIONES DE ANÁLISIS DE SENTIMIENTOS
# ==========================================

def binary_sentiment(df, rating_col='overall', threshold=3):
    """
    Crea etiquetas binarias de sentimiento basadas en rating
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con reviews
    rating_col : str
        Nombre de la columna con ratings
    threshold : int
        Umbral para clasificación (>threshold = positivo)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con columna 'sentiment' añadida
    """
    df = df.copy()
    df['sentiment'] = (df[rating_col] > threshold).astype(int)
    df['sentiment_label'] = df['sentiment'].map({0: 'negative', 1: 'positive'})
    
    return df

def analyze_sentiment_distribution(df):
    """
    Analiza y visualiza la distribución de sentimientos
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con columna 'sentiment_label'
        
    Returns:
    --------
    dict
        Estadísticas de sentimientos
    """
    sentiment_counts = df['sentiment_label'].value_counts()
    total = len(df)
    sentiment_stats = {}
    
    print("Distribución de sentimientos:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        sentiment_stats[sentiment] = {
            'count': count,
            'percentage': percentage
        }
        print(f"  {sentiment}: {count:,} reviews ({percentage:.1f}%)")
    
    # Visualización con porcentajes encima
    ax = sentiment_counts.plot(kind='bar', title='Distribución de Sentimientos', 
                              figsize=(8, 4), color=['#5AA582','#EA5947'])
    
    # Añadir porcentajes encima de las barras
    for i, (sentiment, count) in enumerate(sentiment_counts.items()):
        percentage = sentiment_stats[sentiment]['percentage']
        ax.text(i, count + 200, f'{percentage:.1f}%', ha='center', va='bottom', 
               fontweight='bold', color='#4A4A4A')
    
    plt.xticks(rotation=0)
    plt.ylim(0, max(sentiment_counts) * 1.15)
    plt.show()
    
    return sentiment_stats

# ==========================================
#  FUNCIÓN DE VISUALIZACIÓN
# ==========================================

def plot_word_embeddings(base_words, embeddings_2d, word_clusters, vocab_size, title="Word Embeddings Visualization"):
    """
    Visualiza word embeddings en 2D con clusters por palabra base
    
    Parameters:
    -----------
    base_words : list
        Lista de palabras clave
    embeddings_2d : numpy.array
        Embeddings reducidos a 2D (shape: n_words, n_similar, 2)
    word_clusters : list
        Lista de listas con palabras similares por cada palabra base
    vocab_size : int
        Tamaño del vocabulario para mostrar en el título
    title : str
        Título del gráfico
    """
    plt.figure(figsize=(16, 10))
    
    # Colores distintivos para cada cluster
    colors = plt.cm.Set1(np.linspace(0, 1, len(base_words)))
    
    for i, (base_word, embeddings, words, color) in enumerate(
        zip(base_words, embeddings_2d, word_clusters, colors)
    ):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        
        # Plotear cluster
        plt.scatter(x, y, c=[color], alpha=0.7, s=100, 
                   label=f'"{base_word}" cluster', edgecolors='white', linewidth=1)
        
        # Añadir etiquetas
        for j, word in enumerate(words):
            plt.annotate(word, 
                        xy=(x[j], y[j]), 
                        xytext=(5, 5), 
                        textcoords='offset points',
                        ha='left', va='bottom', 
                        fontsize=9, alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.2', 
                                facecolor='white', alpha=0.7))
    
    # Personalizar gráfico
    plt.title(f'{title}\n(Vocabulario: {vocab_size:,} palabras)', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Dimensión t-SNE 1', fontsize=12)
    plt.ylabel('Dimensión t-SNE 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# ==========================================
# FUNCIONES DE PREPROCESADO 
# ==========================================


def validate_and_convert_text(text: Union[str, float, None]) -> str:
    """
    PASO 0: Verificación robusta de tipo y manejo de valores nulos
    
    Parameters:
    -----------
    text : Union[str, float, None]
        Texto de entrada (cualquier tipo)
        
    Returns:
    --------
    str
        Texto válido o string vacío
    """
    # Manejar casos nulos
    if pd.isna(text) or text is None:
        return ""
    
    # Convertir a string y limpiar
    text = str(text).strip()
    
    # Verificar contenido útil
    if len(text) == 0:
        return ""
        
    return text


def normalize_text(text: str) -> str:
    """
    PASO 1: Normalización avanzada del texto
    
    Parameters:
    -----------
    text : str
        Texto a normalizar
        
    Returns:
    --------
    str
        Texto normalizado
    """
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Eliminar acentos y caracteres especiales
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Normalizar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def remove_punctuation_and_numbers(text: str, remove_numbers: bool = True) -> str:
    """
    PASO 2: Eliminación de puntuación y números
    
    Parameters:
    -----------
    text : str
        Texto normalizado
    remove_numbers : bool
        Si eliminar números o mantenerlos
        
    Returns:
    --------
    str
        Texto sin puntuación y números (opcional)
    """
    # Eliminar números si está activado
    if remove_numbers:
        text = re.sub(r"\d+", "", text)
    
    # Eliminar puntuación usando translate (más eficiente)
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Limpiar espacios resultantes
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text: str, 
                 tokenizer: Optional[RegexpTokenizer] = None, 
                 method: str = "regex") -> List[str]:
    """
    PASO 3: Tokenización optimizada con inyección de dependencia
    
    Parameters:
    -----------
    text : str
        Texto limpio
    tokenizer : Optional[RegexpTokenizer]
        Tokenizer reutilizable para mejor rendimiento
    method : str
        Método de tokenización: 'simple' o 'regex'
        
    Returns:
    --------
    List[str]
        Lista de tokens
    """
    if method == "simple":
        # Método rápido
        return text.split()
    
    # Método robusto con inyección de dependencia
    if tokenizer is None:
        tokenizer = RegexpTokenizer(r'\w+')
    
    return tokenizer.tokenize(text)


def filter_tokens(tokens: List[str], 
                 remove_stopwords: bool = True,
                 custom_stopwords: Optional[List[str]] = None,
                 min_length: int = 2) -> List[str]:
    """
    PASO 4: Filtrado avanzado de tokens
    
    Parameters:
    -----------
    tokens : List[str]
        Lista de tokens a filtrar
    remove_stopwords : bool
        Eliminar stopwords
    custom_stopwords : Optional[List[str]]
        Stopwords personalizadas
    min_length : int
        Longitud mínima de tokens
        
    Returns:
    --------
    List[str]
        Lista de tokens filtrados
    """
    if not remove_stopwords:
        # Solo filtrar por longitud
        return [token for token in tokens if len(token) >= min_length]
    
    # Configurar stopwords
    if custom_stopwords is not None:
        stop_words = set(custom_stopwords)
    else:
        stop_words = set(stopwords.words("english"))
    
    # Filtrado combinado
    filtered_tokens = []
    for token in tokens:
        # Filtrar stopwords
        if token.lower() in stop_words:
            continue
            
        # Filtrar por longitud mínima
        if len(token) < min_length:
            continue
            
        # Filtrar tokens que son solo puntuación residual
        if token in string.punctuation:
            continue
            
        filtered_tokens.append(token)
    
    return filtered_tokens


def lemmatize_tokens(tokens: List[str], 
                    lemmatizer: Optional[WordNetLemmatizer] = None,
                    advanced: bool = True) -> List[str]:
    """
    PASO 5: Lematización optimizada con inyección de dependencia
    
    Parameters:
    -----------
    tokens : List[str]
        Lista de tokens a lematizar
    lemmatizer : Optional[WordNetLemmatizer]
        Lemmatizer reutilizable para mejor rendimiento
    advanced : bool
        Si usar lematización multi-POS (True) o simple (False)
        
    Returns:
    --------
    List[str]
        Lista de tokens lematizados
    """
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    
    lemmatized_tokens = []
    
    for token in tokens:
        if advanced:
            # Método avanzado multi-POS
            lemma = lemmatizer.lemmatize(token, pos='v')  # Verbo
            if lemma == token:
                lemma = lemmatizer.lemmatize(token, pos='n')  # Sustantivo
            if lemma == token:
                lemma = lemmatizer.lemmatize(token, pos='a')  # Adjetivo
        else:
            # Método simple
            lemma = lemmatizer.lemmatize(token)
            
        lemmatized_tokens.append(lemma)
    
    return lemmatized_tokens


def preprocess_text(text: Union[str, float, None],
                   remove_stopwords: bool = True,
                   lemmatize: bool = True,
                   remove_numbers: bool = True,
                   custom_stopwords: Optional[List[str]] = None,
                   min_length: int = 2,
                   tokenize_method: str = "regex",
                   advanced_lemma: bool = True,
                   tokenizer: Optional[RegexpTokenizer] = None,
                   lemmatizer: Optional[WordNetLemmatizer] = None,
                   debug: bool = False) -> str:
    """
    FUNCIÓN PRINCIPAL OPTIMIZADA: Pipeline completo de preprocesado
    
    Parameters:
    -----------
    text : Union[str, float, None]
        Texto de entrada (manejo robusto de tipos)
    remove_stopwords : bool
        Eliminar stopwords
    lemmatize : bool
        Aplicar lematización
    remove_numbers : bool
        Eliminar números
    custom_stopwords : Optional[List[str]]
        Stopwords personalizadas
    min_length : int
        Longitud mínima de tokens
    tokenize_method : str
        Método de tokenización ('simple' o 'regex')
    advanced_lemma : bool
        Lematización avanzada (multi-POS) o simple
    tokenizer : Optional[RegexpTokenizer]
        Tokenizer reutilizable para mejor rendimiento
    lemmatizer : Optional[WordNetLemmatizer]
        Lemmatizer reutilizable para mejor rendimiento
    debug : bool
        Mostrar información de debug
        
    Returns:
    --------
    str
        Texto completamente procesado
    """
    # PASO 0: Validación robusta
    text = validate_and_convert_text(text)
    if not text:
        return ""
    
    # PASO 1: Normalización avanzada
    text = normalize_text(text)
    
    # PASO 2: Limpieza de puntuación y números
    text = remove_punctuation_and_numbers(text, remove_numbers)
    
    # PASO 3: Tokenización optimizada
    tokens = tokenize_text(text, tokenizer, tokenize_method)
    
    # PASO 4: Filtrado avanzado
    tokens = filter_tokens(tokens, remove_stopwords, custom_stopwords, min_length)
    
    # PASO 5: Lematización configurable
    if lemmatize:
        tokens = lemmatize_tokens(tokens, lemmatizer, advanced_lemma)
    
    # Debug opcional
    if debug and tokens:
        print(f"[DEBUG] {len(tokens)} tokens generados: {tokens[:5]}{'...' if len(tokens) > 5 else ''}")
    
    # Reconstrucción final
    return " ".join(tokens)


def preprocess_corpus(df: pd.DataFrame, 
                     text_column: str = 'reviewText', 
                     new_column: str = 'processed_text',
                     show_sample: bool = True,
                     **kwargs) -> pd.DataFrame:
    """
    Aplica preprocesado optimizado a todo un corpus con objetos reutilizables
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con las reviews
    text_column : str
        Nombre de la columna con texto original
    new_column : str
        Nombre de la nueva columna con texto procesado
    show_sample : bool
        Mostrar muestra de resultados antes/después
    **kwargs : dict
        Parámetros adicionales para preprocess_text()
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con nueva columna de texto procesado
    """
    print(f"Iniciando preprocesado de {len(df)} reviews...")
    
    # Crear objetos reutilizables para mejor rendimiento
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    
    # Aplicar preprocesado optimizado con objetos reutilizables
    df[new_column] = df[text_column].apply(
        lambda x: preprocess_text(x, tokenizer=tokenizer, lemmatizer=lemmatizer, **kwargs)
    )
    
    # Estadísticas detalladas
    original_words = df[text_column].fillna('').astype(str).str.split().str.len().sum()
    processed_words = df[new_column].str.split().str.len().sum()
    empty_after_processing = (df[new_column] == "").sum()
    
    print(f"Preprocesado completado:")
    print(f"Palabras: {original_words:,} → {processed_words:,}")
    print(f"Reducción: {((original_words - processed_words) / original_words * 100):.1f}%")
    print(f"Reviews vacías tras procesado: {empty_after_processing}")
    
    # Mostrar muestra de resultados
    if show_sample and len(df) > 0:
        print("\nMuestra de resultados:")
        sample_df = df[[text_column, new_column]].sample(min(3, len(df)))
        for i, (_, row) in enumerate(sample_df.iterrows(), 1):
            original = str(row[text_column])
            processed = str(row[new_column])
            print(f"\n{i}. Original: {original[:150]}{'...' if len(original) > 150 else ''}")
            print(f"   Procesado: {processed}")
    
    return df