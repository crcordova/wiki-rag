# Wiki RAG App

Wiki RAG App es una aplicación interactiva desarrollada con Streamlit que permite realizar consultas sobre contenidos de Wikipedia utilizando una arquitectura de RAG (Retrieval-Augmented Generation) y modelos de lenguaje (LLMs). El sistema es capaz de recuperar contexto relevante desde un índice vectorial y generar respuestas con modelos como Groq (LLaMA 3), OpenAI GPT-3.5 o Ollama, según configuración.

## Características
 - Recuperación de contexto desde páginas de Wikipedia  

 - Generación de respuestas mediante modelos LLM  

 - Indexación de documentos mediante embeddings (instructor-large)  

 - Opción de actualizar el índice dinámicamente  

 - Compatibilidad con múltiples proveedores de LLM (Groq, OpenAI, Ollama)  

 ## Arquitectura

 - RAG (Retrieval-Augmented Generation): combinación de recuperación vectorial y generación de lenguaje.  

 - Vector Store: construido con VectorStoreIndex de llama-index, persistido localmente.  

 - Embedding Model: hkunlp/instructor-large vía HuggingFace.  

 - LLM: por defecto utiliza Groq con el modelo llama3-8b-8192, pero puede modificarse para usar OpenAI o Ollama.  

 - Fuente de datos: Páginas de Wikipedia definidas en pages.py.  

 ## Uso
 ### Clonar repositorio

```bash
 git clone https://github.com/tuusuario/wiki-rag-app.git  
 cd wiki-rag-app
```
 ### Instalar dependencias

```bash
 pip install -r requirements.txt
```

 ### Configura tus claves de API
```bash
 export GROQ_API_KEY=tu_clave
```
o bien copiando el file `.env`
```bash
cp .env.example .env
```

### Ejecuta la aplicación
```bash
 streamlit run app.py
```

Para cambiar las páginas indexadas, modifica el archivo `pages.py`:
```bash
 PAGES = ["Inteligencia_artificial", "Aprendizaje_automático", "Modelo_de_lenguaje"]
```  

Para cambiar el modelo LLM, edita la configuración en las funciones `get_index()` o `get_query_engine()`.

## Requisitos
 - Python 3.10+  

 - Conexión a internet para acceder a Wikipedia y los LLMs  