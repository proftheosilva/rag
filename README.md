# RAG didático — Cucando TI

Aplicação educacional para acompanhar a ingestão de documentos estruturados e não-estruturados, tratamento em **Arquitetura Medalhão (Raw ➔ Silver ➔ Gold)** e consulta com **Retrieval-Augmented Generation (RAG)** integrado a um banco analítico e vetorial local com **DuckDB**. 

A interface apresenta a extração polimórfica, a divisão em trechos, os embeddings, a busca vetorial por similaridade de cosseno, a inspeção analítica e o contexto enviado à LLM.

---

## Como funciona

### 1. Ingestão e Arquitetura Medalhão

O pipeline segue o padrão Medalhão com persistência em arquivos colunares Parquet e banco analítico/vetorial DuckDB:

```text
Entrada (RAW / Subpastas ou Webhook)
       │
       ▼
Identificação de Tipo & Extração Polimórfica
 (Estruturado: CSV/JSON/SQL  │  Não Estruturado: PDF)
       │
       ▼
Camada SILVER
 - Limpeza e higienização
 - Descarte de registros inconsistentes e nulos críticos
 - Concatenação incremental em Parquet por categoria (ex: notas_fiscais.parquet)
       │
       ▼
Camada GOLD
 - Parquet consolidado promovido e pronto para Lakehouse / Data Warehouse
       │
       ▼
Banco Analítico & Vetorial DUCKDB (dados/documentos/database/analytics.duckdb)
 ├── Tabelas Analíticas: sincronizadas diretamente do Parquet (ex: CREATE TABLE notas_fiscais AS SELECT * FROM read_parquet(...))
 ├── Tabela Relacional: 'documentos' (metadados, status e observabilidade de extração)
 └── Tabela Vetorial: 'documento_chunks' (chunks de texto + vetor FLOAT[384] gerado pelo FastEmbed)
```

1. **Recepção / Varredura RAW:**
   - Documentos organizados por seções/divisões empresariais dentro de `dados/documentos/raw` (exemplo didático: subpasta `dados/documentos/raw/FINANCEIRO/`).
   - Também suporta recebimento direto via **Webhook** (upload binário multipart, base64 ou URL remota em `/api/webhook` e `/api/ingest`).
2. **Extração Polimórfica:**
   - **Estruturados e Semiestruturados (`.csv`, `.json`, `.sql`):** extração e tabulação automática de registros via Pandas.
   - **Não-estruturados (`.pdf`, texto):** extração de texto integral via `pdfplumber`/`pypdf` e extração de campos semânticos por expressões regulares.
3. **Tratamento na Camada Silver:**
   - Mapeamento e tipagem dos campos.
   - Higienização e remoção de registros com nulos críticos para consistência.
   - Adição incremental ao arquivo Parquet por categoria (ex: `silver/notas_fiscais.parquet`), unindo novos registros sem criar arquivos avulsos fragmentados.
4. **Consolidação na Camada Gold:**
   - O Parquet final e higienizado é promovido para a pasta `gold/`.
5. **Persistência Analítica e Vetorial no DuckDB:**
   - O arquivo `dados/documentos/database/analytics.duckdb` mantém:
     - As tabelas analíticas tabulares geradas dos Parquets (ex: tabela `notas_fiscais`).
     - A tabela relacional `documentos` (histórico, total de campos, preenchidos, nulos, percentual de sucesso e status).
     - A tabela vetorial `documento_chunks` (armazenando o texto do chunk, metadados em JSON e vetor numérico `FLOAT[]` de 384 dimensões).

---

### 2. Consulta RAG com DuckDB Vetorial

```text
Pergunta do Usuário ──► Tokenização (tiktoken) ──► FastEmbed (Vetor 384d)
                                                           │
                                                           ▼
                                         Busca Vetorial Nativa no DuckDB
                                         list_cosine_similarity(vetor, query)
                                                           │
                                                           ▼
                                          Trechos Relevantes Recuperados
                                                           │
                                                           ▼
Prompt Aumentado (Instruções + Contexto Top-K + Pergunta) ──► OpenRouter (LLM) ──► Resposta Final
```

1. **Tokenização:** A pergunta é inspecionada didaticamente com `tiktoken` (`cl100k_base`).
2. **Embedding:** Geração do vetor denso de 384 dimensões com o modelo local `BAAI/bge-small-en-v1.5` via FastEmbed.
3. **Retrieval Vetorial no DuckDB:**
   - Substituição do Supabase/pgvector pela função de similaridade nativa do DuckDB:
     ```sql
     SELECT id, documento_id, texto, metadados,
            list_cosine_similarity(vetor, ?::FLOAT[]) AS similarity
     FROM documento_chunks
     WHERE list_cosine_similarity(vetor, ?::FLOAT[]) >= ?
     ORDER BY similarity DESC
     LIMIT ?;
     ```
4. **Regra Estrita de Ausência de Dados:**
   - Se nenhum trecho atingir o limiar mínimo de relevância (`min_threshold`), o sistema retorna imediatamente `Sem dados disponiveis na base` sem gastar tokens nem acionar a LLM.
5. **Geração Contextualizada:**
   - Havendo evidências relevantes, o prompt aumentado é montado e submetido à LLM via OpenRouter (`openrouter/auto`), instruindo o modelo a responder estritamente com base nos documentos citados.

---

## Tecnologias

- **Frontend:** Next.js 16, React 19, JavaScript e CSS Vanilla; encaminha chamadas `/api/*` ao backend e exibe o status de DuckDB, OpenRouter e pipeline medalhão.
- **Backend:** Python, FastAPI, Uvicorn, Pydantic e HTTPX.
- **Banco Analítico e Vetorial:** **DuckDB** (substituindo Supabase PGVector para metadados relacionais, tabelas analíticas e busca vetorial por similaridade de cosseno embutida).
- **Formatos de Dados & Lakehouse:** **Apache Parquet** (`pyarrow` e `pandas`) particionado por tipo nas camadas `silver/` e `gold/`.
- **Extração de Documentos:** `pdfplumber`, `pypdf`, `pandas` (CSV/JSON/SQL) e regex customizado.
- **Embeddings:** FastEmbed local (ONNX Runtime), modelo `BAAI/bge-small-en-v1.5` (384 dimensões).
- **Tokenização Didática:** `tiktoken`.
- **Geração (LLM):** API OpenRouter (`openrouter/auto`).
- **Ambiente:** Execução local ou via Docker Desktop / Docker Compose.

---

## Estrutura de Diretórios

O projeto segue a padronização centralizada na pasta `dados/documentos` na raiz do laboratório:

```text
laboratorio-ia-docker/
├── compose.yaml
├── apps/
│   └── rag/                              # Aplicação RAG
│       ├── .env                          # Variáveis de ambiente
│       ├── Dockerfile                    # Container Python do backend
│       ├── backend/
│       │   ├── config.py                 # Mapeamento das camadas e caminhos
│       │   ├── database.py               # DuckDBManager (relacional, vetorial e analítico)
│       │   ├── extractor.py              # Extrator polimórfico (PDF, CSV, JSON, SQL)
│       │   ├── medallion_service.py      # Gestor das camadas Raw, Silver e Gold (Parquet)
│       │   ├── rag_service.py            # Orquestrador do pipeline RAG e varredura
│       │   ├── chunker.py                # Divisão semântica por blocos
│       │   ├── embedder.py               # Geração local de embeddings (FastEmbed)
│       │   ├── tokenizer.py              # Tokenizador didático (tiktoken)
│       │   ├── test_pipeline.py          # Script de testes automatizados ponta a ponta
│       │   └── main.py                   # API FastAPI e rotas de webhook/ingestão/consulta
│       ├── frontend/
│       └── requirements.txt
└── dados/
    └── documentos/                       # Pasta compartilhada do Lakehouse / Documentos
        ├── raw/                          # Camada RAW dividida por divisões da empresa
        │   ├── FINANCEIRO/               # Exemplo didático: subpasta do setor financeiro
        │   ├── RH/
        │   ├── PAGAMENTO/
        │   └── ALMOXARIFADO/
        ├── silver/                       # Camada SILVER: dados tratados em .parquet por tipo
        │   └── notas_fiscais.parquet
        ├── gold/                         # Camada GOLD: parquets consolidados prontos para analytics
        │   └── notas_fiscais.parquet
        └── database/                     # Banco Analítico e Vetorial Local
            └── analytics.duckdb          # Arquivo do banco DuckDB
```

---

## Configuração das Variáveis de Ambiente (`.env`)

Crie o arquivo `apps/rag/.env`:

```dotenv
# Provedor LLM
OPENROUTER_API_KEY=sua_chave_aqui
OPENROUTER_MODEL=openrouter/auto

# Configurações do RAG e Embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# Origem de Ingestão Padrão ('raw' ou 'webhook')
DOCUMENT_SOURCE=raw

# Opcional: Caminhos personalizados (caso deseje sobrescrever o padrão)
# BASE_DOCS_DIR=/caminho/para/dados/documentos
# DUCKDB_PATH=/caminho/para/dados/documentos/database/analytics.duckdb
```

> **Nota sobre o Supabase:** O Supabase foi substituído pelo **DuckDB** para persistência e busca vetorial local de alta velocidade. As chaves do Supabase não são mais obrigatórias para a execução do pipeline.

---

## Execução Local e Docker

### Opção A: Execução via Docker Compose (Recomendado no Laboratório)

Na raiz do laboratório:

```powershell
docker compose up -d --build rag rag-frontend
```

Acompanhar logs:
```powershell
docker compose logs -f rag rag-frontend
```

Acessos:
- **Aplicação Web:** http://localhost:3000
- **Documentação Swagger (FastAPI):** http://localhost:8000/docs
- **Status do Sistema:** http://localhost:8000/api/status

---

### Opção B: Execução Local no Terminal

1. Instalar dependências:
   ```powershell
   cd apps/rag
   pip install -r requirements.txt
   ```

2. Executar testes automatizados do pipeline:
   ```powershell
   python -m backend.test_pipeline
   ```

3. Iniciar o servidor backend:
   ```powershell
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

| Método e Rota | Finalidade |
| --- | --- |
| `GET /api/status` | Exibe estado do DuckDB, OpenRouter, dimensões do FastEmbed e arquitetura. |
| `GET /api/available-files?secao=FINANCEIRO` | Varre a subpasta correspondente em `raw` e compara com os arquivos indexados no DuckDB. |
| `GET /api/documents` | Lista os documentos cadastrados na tabela relacional `documentos` do DuckDB. |
| `GET /api/duckdb-tables` | Lista todas as tabelas analíticas e vetoriais existentes no `analytics.duckdb`. |
| `GET /api/catalogo` | Retorna o dicionário de dados corporativo extraído pelo dbt/DuckDB das tabelas em Gold. |
| `POST /api/ingest` | Processa arquivos da subpasta informada (ex: `FINANCEIRO`) sob a arquitetura Medalhão (Silver ➔ Gold ➔ DuckDB). |
| `POST /api/webhook` | Endpoint dedicado para recebimento externo (n8n, ERP, etc.) via multipart, JSON base64 ou binário puro. |
| `POST /api/query` | Executa a busca vetorial por similaridade de cosseno no DuckDB e responde via LLM. |
| `DELETE /api/documents` | Limpa as tabelas de chunks e documentos no DuckDB para reinício de testes. |

---

## Catálogo de Dados Corporativo (NocoDB) e dbt-duckdb

O pipeline conta com **dbt-duckdb** e **NocoDB** para transformar e catalogar os dados:
- **dbt-duckdb (`apps/rag/dbt_project`):** 
  - Lê diretamente os arquivos `.parquet` em `gold/` via motor colunar do DuckDB.
  - Executa modelos declarativos em SQL com testes de qualidade (`not_null`, `unique`).
  - Extrai o esquema e tipos de dados do `information_schema.columns` gerando a tabela `catalogo_metadados`.
- **NocoDB (`laboratorio-nocodb` na porta 8085):**
  - Fornece aos alunos e equipes corporativas uma interface visual em tempo real (como Airtable/Smart Spreadsheet) para navegar pelas tabelas analíticas e consultar o Dicionário de Dados.
  - Acessível em: **`http://localhost:8085`**


---

## Integração com n8n e Webhooks

Para disparar a ingestão via n8n:
- **Via nome de arquivo em subpasta:**
  ```json
  POST http://rag:8000/api/ingest
  {
    "filename": "NF_8010_AuditContabil.pdf",
    "secao": "FINANCEIRO"
  }
  ```
- **Via envio direto de arquivo (sem depender da pasta física):**
  - Enviar requisição para `http://rag:8000/api/webhook` usando `multipart/form-data` com o campo `file` contendo o arquivo binário ou enviando JSON com `file_base64`.
