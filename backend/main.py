import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

from backend.config import settings
from backend.database import db
from backend.extractor import extractor
from backend.rag_service import rag_service
from backend.medallion_service import medallion_service
from backend.chunker import chunker
from backend.tokenizer import tokenizer
from backend.embedder import embedder

app = FastAPI(
    title="RAG Didático API - Teophilo",
    description="API com Arquitetura Medalhão (Raw, Silver, Gold) e DuckDB para RAG Vetorial e Analítico",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    pergunta: str
    top_k: int = Field(default=5, ge=1, le=20)
    min_threshold: float = Field(default=0.35, ge=0, le=1)

class IngestRequest(BaseModel):
    filename: Optional[str] = None
    origem: Optional[str] = None  # "raw" ou "webhook"
    secao: Optional[str] = "FINANCEIRO"
    file_base64: Optional[str] = None
    file_url: Optional[str] = None

@app.get("/api/status")
def get_status():
    duckdb_ok = db.check_connection()
    openrouter_configured = bool(settings.OPENROUTER_API_KEY)
    
    return {
        "status": "online",
        "services": {
            "duckdb": "conectado" if duckdb_ok else "falha_conexao",
            "openrouter": "configurado" if openrouter_configured else "nao_configurado",
            "fastembed_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "arquitetura": "Medalhao (Raw -> Silver -> Gold -> DuckDB)",
            "database_path": settings.DUCKDB_PATH
        }
    }

@app.get("/api/available-files")
def list_available_files(secao: Optional[str] = "FINANCEIRO"):
    """
    Varre as subpastas da pasta raw (ex: FINANCEIRO) listando estritamente os
    arquivos físicos disponíveis para ingestão (Responsabilidade Única / Camada RAW).
    """
    scanned = rag_service.scan_raw_files(subfolder=secao)
    local_files = [item["relative_path"] for item in scanned]
    filenames = [item["filename"] for item in scanned]

    return {
        "secao": secao,
        "total_arquivos": len(local_files),
        "filenames": sorted(filenames),
        "files": sorted(local_files),
        "detalhes": scanned
    }

@app.get("/api/documents/summary")
def get_documents_summary():
    """
    Endpoint dedicado de Governança e Auditoria do DuckDB:
    Consulta o estado consolidado das tabelas de controle e vetores.
    """
    try:
        return db.get_ingestion_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar resumo do DuckDB: {str(e)}")

@app.get("/api/documents")
def list_documents():
    docs = db.list_documentos()
    return {"documentos": docs}

@app.get("/api/duckdb-tables")
def list_duckdb_tables():
    """Lista tabelas criadas no banco analítico DuckDB."""
    try:
        with db.get_connection() as con:
            tables = con.execute("SHOW TABLES;").fetchall()
            result = [t[0] for t in tables]
            return {"tabelas": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao listar tabelas do DuckDB: {str(e)}")

@app.get("/api/catalogo")
def get_data_catalog():
    """
    Retorna o dicionário de dados corporativo gerado pelo dbt/DuckDB
    a partir das tabelas consolidadas da camada Gold em Parquet.
    """
    try:
        with db.get_connection() as con:
            tables = [t[0] for t in con.execute("SHOW TABLES;").fetchall()]
            if "catalogo_metadados" not in tables:
                from backend.medallion_service import medallion_service
                medallion_service.run_dbt_and_update_catalog()
            df = con.execute("SELECT * FROM catalogo_metadados ORDER BY tabela, posicao_ordem;").fetchdf()
            if df.empty:
                return {"total_campos": 0, "catalogo": []}
            df["sincronizado_em"] = df["sincronizado_em"].astype(str)
            return {
                "total_campos": len(df),
                "catalogo": df.to_dict(orient="records"),
                "nocodb_url": "http://localhost:8080"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao consultar catálogo de dados: {str(e)}")


@app.post("/api/ingest")
async def ingest_documents(request: Request):
    """
    Endpoint de ingestão flexível sob arquitetura medalhão:
    1. Se vier via Webhook (multipart, base64 ou binário puro): processa diretamente em memória.
    2. Se vier via Pasta Raw: varre as subpastas (ex: FINANCEIRO), higieniza em Silver (Parquet),
       promove para Gold, sincroniza no DuckDB analítico e popula o DuckDB vetorial.
    """
    content_type = request.headers.get("content-type", "").lower()

    # 1. Binário puro via Webhook
    if "application/pdf" in content_type or "application/octet-stream" in content_type:
        file_bytes = await request.body()
        if file_bytes:
            pdf_filename = (
                request.headers.get("x-filename")
                or request.query_params.get("filename")
            )
            if not pdf_filename:
                pdf_filename = f"doc_webhook_{uuid.uuid4().hex[:8]}.pdf"

            try:
                res = rag_service.process_and_ingest_document(
                    pdf_filename=pdf_filename,
                    file_bytes=file_bytes,
                    from_webhook=True,
                    source="webhook"
                )
                return {
                    "total_processados": 1,
                    "detalhes": [res],
                    "mensagem": f"Documento recebido via Webhook ({content_type}) e processado com sucesso na camada Medalhão + DuckDB.",
                    "arquivos_inseridos": [res["nome_arquivo"]],
                    "silver_parquet": res.get("silver_parquet"),
                    "gold_parquet": res.get("gold_parquet")
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Erro ao processar binário via Webhook: {str(e)}")

    # 2. Multipart/Form-Data
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload_file = form.get("file") or form.get("data") or form.get("documento")
        file_bytes = None
        pdf_filename = None
        if upload_file and hasattr(upload_file, "read"):
            file_bytes = await upload_file.read()
            pdf_filename = getattr(upload_file, "filename", None)
        if not pdf_filename and "filename" in form:
            pdf_filename = str(form.get("filename"))
        
        if file_bytes:
            try:
                res = rag_service.process_and_ingest_document(
                    pdf_filename=pdf_filename,
                    file_bytes=file_bytes,
                    from_webhook=True,
                    source="webhook"
                )
                return {
                    "total_processados": 1,
                    "detalhes": [res],
                    "mensagem": f"Documento recebido via Webhook (multipart) e processado: {res['nome_arquivo']}.",
                    "arquivos_inseridos": [res["nome_arquivo"]],
                    "silver_parquet": res.get("silver_parquet"),
                    "gold_parquet": res.get("gold_parquet")
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Erro ao processar arquivo via upload: {str(e)}")

    # 3. Requisição JSON (Webhook ou Disparo de Ingestão)
    body = {}
    if "application/json" in content_type or not content_type:
        try:
            body = await request.json()
        except Exception:
            body = {}

    req_filename = body.get("filename")
    req_origem = body.get("origem")
    req_secao = body.get("secao") or "FINANCEIRO"
    req_file_base64 = body.get("file_base64") or body.get("data")
    req_file_url = body.get("file_url") or body.get("url")

    # Ingestão via Webhook por JSON
    if req_origem == "webhook" or req_file_base64 or req_file_url:
        try:
            res = rag_service.process_and_ingest_document(
                pdf_filename=req_filename,
                file_base64=req_file_base64,
                file_url=req_file_url,
                from_webhook=True,
                source="webhook"
            )
            return {
                "total_processados": 1,
                "detalhes": [res],
                "mensagem": f"Documento recebido via Webhook e processado com sucesso: {res['nome_arquivo']}.",
                "arquivos_inseridos": [res["nome_arquivo"]],
                "silver_parquet": res.get("silver_parquet"),
                "gold_parquet": res.get("gold_parquet")
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Erro ao processar documento via Webhook: {str(e)}")

    # Ingestão a partir da Pasta RAW (subpastas ex: FINANCEIRO)
    try:
        summary = db.get_ingestion_summary()
    except Exception:
        raise HTTPException(status_code=503, detail="Não foi possível consultar a ingestão no DuckDB.")

    indexed_names = set(summary["arquivos_indexados"])

    if req_filename:
        files_to_process = [req_filename]
    else:
        scanned = rag_service.scan_raw_files(subfolder=req_secao)
        files_to_process = [item["filename"] for item in scanned]

    if not files_to_process:
        raise HTTPException(status_code=400, detail=f"Nenhum arquivo encontrado para ingestão na pasta raw/{req_secao}.")

    results = []
    processing_attempts = 0
    for fn in files_to_process:
        processing_attempts += 1
        try:
            res = rag_service.process_and_ingest_document(
                pdf_filename=fn,
                source="raw",
                subfolder=req_secao
            )
            results.append(res)
        except Exception as e:
            results.append({
                "nome_arquivo": fn,
                "status": "ERRO",
                "erro": str(e)
            })

    updated_summary = db.get_ingestion_summary()
    inserted_files = [
        r["nome_arquivo"] for r in results if r.get("status") == "CONCLUIDO"
    ]
    failed_files = [
        r["nome_arquivo"] for r in results if r.get("status") == "ERRO"
    ]

    message = f"Processamento concluído na Arquitetura Medalhão + DuckDB. Inseridos: {len(inserted_files)}."
    if failed_files:
        message += f" Falhas em: {', '.join(failed_files)}."

    return {
        "total_processados": processing_attempts,
        "detalhes": results,
        "mensagem": message,
        "arquivos_inseridos": inserted_files,
        "resumo_atualizado": True,
        **updated_summary,
    }

@app.post("/api/query")
def execute_query(req: QueryRequest):
    if not req.pergunta.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    
    try:
        resultado = rag_service.query_rag(
            question=req.pergunta,
            top_k=req.top_k,
            min_threshold=req.min_threshold
        )
        return resultado
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar consulta: {str(e)}")

@app.delete("/api/documents")
def clear_documents():
    try:
        db.clear_all()
        return {"mensagem": "Todos os documentos e chunks foram removidos com sucesso do DuckDB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao limpar banco: {str(e)}")

@app.post("/api/webhook")
async def webhook_endpoint(request: Request):
    """
    Endpoint dedicado para recebimento de documentos via Webhook.
    Processa diretamente na Arquitetura Medalhão e persiste no DuckDB.
    """
    try:
        content_type = request.headers.get("content-type", "").lower()
        pdf_filename = None
        file_bytes = None
        file_base64 = None
        file_url = None

        if "multipart/form-data" in content_type:
            form = await request.form()
            upload_file = form.get("file") or form.get("data") or form.get("documento")
            if upload_file and hasattr(upload_file, "read"):
                file_bytes = await upload_file.read()
                pdf_filename = getattr(upload_file, "filename", None)
            if not pdf_filename and "filename" in form:
                pdf_filename = str(form.get("filename"))
            if not file_bytes and "file_base64" in form:
                file_base64 = str(form.get("file_base64"))
            if "file_url" in form:
                file_url = str(form.get("file_url"))

        elif "application/json" in content_type:
            body = await request.json()
            pdf_filename = body.get("filename")
            file_base64 = body.get("file_base64") or body.get("data")
            file_url = body.get("file_url") or body.get("url")

        elif "application/pdf" in content_type or "application/octet-stream" in content_type:
            file_bytes = await request.body()
            pdf_filename = request.headers.get("x-filename") or request.query_params.get("filename")

        else:
            try:
                body = await request.json()
                pdf_filename = body.get("filename")
                file_base64 = body.get("file_base64") or body.get("data")
                file_url = body.get("file_url") or body.get("url")
            except Exception:
                file_bytes = await request.body()
                pdf_filename = request.headers.get("x-filename") or request.query_params.get("filename")

        resultado = rag_service.process_and_ingest_document(
            pdf_filename=pdf_filename,
            file_bytes=file_bytes,
            file_base64=file_base64,
            file_url=file_url,
            from_webhook=True,
            source="webhook"
        )

        return {
            "status": "sucesso",
            "origem": "webhook",
            "mensagem": f"Documento recebido via Webhook e processado com sucesso: {resultado['nome_arquivo']}.",
            "resultado": resultado
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento via Webhook: {str(e)}")


# ==============================================================================
# ENDPOINTS MODULARES / STEP-BY-STEP (PARA O WORKFLOW DIDÁTICO NO N8N)
# ==============================================================================

class StepExtractRequest(BaseModel):
    filename: Optional[str] = None
    secao: Optional[str] = "FINANCEIRO"
    source: Optional[str] = "raw"  # "raw" ou "webhook"
    file_base64: Optional[str] = None
    file_url: Optional[str] = None

@app.post("/api/pipeline/extract")
async def step_extract(request: Request):
    """
    PASSO 1 DO N8N: Extração e Categorização
    Recebe o arquivo (RAW ou Webhook), roda o DocumentExtractor e retorna:
    - Categoria identificada (ex: Notas Fiscais)
    - Tabela destino correspondente
    - Campos e valores extraídos com total e métricas de qualidade
    """
    try:
        raw_body = await request.body()
        print(f"[DEBUG EXTRACT] raw_body: {raw_body}, content_type: {request.headers.get('content-type')}")
        body = {}
        if raw_body:
            try:
                body = json.loads(raw_body.decode("utf-8"))
                if isinstance(body, str):
                    body = json.loads(body)
            except Exception as e_json:
                print(f"[DEBUG EXTRACT] erro json parse: {e_json}")
                body = {}

        req_filename = body.get("filename") or request.query_params.get("filename")
        req_secao = body.get("secao") or request.query_params.get("secao") or "FINANCEIRO"
        req_source = body.get("source") or request.query_params.get("source") or "raw"
        req_file_base64 = body.get("file_base64")
        req_file_url = body.get("file_url")

        print(f"[DEBUG EXTRACT] req_filename: {req_filename}, req_secao: {req_secao}, req_source: {req_source}")

        from_webhook = (req_source == "webhook") or bool(req_file_base64 or req_file_url)
        received = rag_service.receive_document(
            pdf_filename=req_filename,
            file_base64=req_file_base64,
            file_url=req_file_url,
            from_webhook=from_webhook,
            source="webhook" if from_webhook else "raw",
            subfolder=req_secao
        )

        filename = received["filename"]
        file_input = received["file_input"]
        file_type = received["file_type"]
        secao = received["secao"]

        if file_type in ["csv", "json", "sql"]:
            records, categoria, table_type = extractor.extract_structured(file_input, filename, file_type)
            raw_text = "\n".join([str(r) for r in records])
            parsed = {
                "nome_arquivo": filename,
                "categoria": categoria,
                "tipo_tabela": table_type,
                "total_campos": len(records[0]) if records else 0,
                "campos_preenchidos": len(records[0]) if records else 0,
                "campos_nulos": 0,
                "percentual_sucesso": 100.0,
                "status": "PROCESSADO",
                "dados": records[0] if records else {}
            }
            records_to_silver = records
        else:
            raw_text = extractor.extract_text_from_pdf(file_input)
            parsed_invoice = extractor.parse_invoice_fields(raw_text, filename)
            table_type = parsed_invoice.get("tipo_tabela", "notas_fiscais")
            categoria = parsed_invoice.get("categoria", "Notas Fiscais")
            records_to_silver = [{
                "nome_arquivo": filename,
                "numero_nota": parsed_invoice.get("numero_nota"),
                "serie": parsed_invoice.get("serie"),
                "data_emissao": parsed_invoice.get("data_emissao"),
                "fornecedor": parsed_invoice.get("fornecedor"),
                "cnpj_fornecedor": parsed_invoice.get("cnpj_fornecedor"),
                "cliente": parsed_invoice.get("cliente"),
                "cnpj_cliente": parsed_invoice.get("cnpj_cliente"),
                "natureza_operacao": parsed_invoice.get("natureza_operacao"),
                "valor_total": parsed_invoice.get("valor_total"),
                "observacoes": parsed_invoice.get("observacoes"),
                "itens": parsed_invoice.get("itens"),
                "secao": secao,
                "status": parsed_invoice.get("status")
            }]
            parsed = parsed_invoice

        return {
            "etapa": "1_extracao",
            "status": "SUCESSO",
            "nome_arquivo": filename,
            "origem": received["origin"],
            "secao": secao,
            "categoria": categoria,
            "tipo_tabela": table_type,
            "metricas": {
                "total_campos": parsed.get("total_campos", 0),
                "campos_preenchidos": parsed.get("campos_preenchidos", 0),
                "campos_nulos": parsed.get("campos_nulos", 0),
                "percentual_sucesso": parsed.get("percentual_sucesso", 100.0)
            },
            "campos_extraidos": parsed,
            "records_para_silver": records_to_silver,
            "raw_text_preview": raw_text[:500] if raw_text else ""
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na Etapa 1 (Extração): {str(e)}")


@app.post("/api/pipeline/silver")
async def step_silver(request: Request):
    """
    PASSO 2 DO N8N: Higienização e Gravação na Camada Silver (.parquet)
    Recebe os registros do Passo 1, higieniza e faz o append incremental no Parquet da categoria.
    """
    try:
        body = await request.json()
        tipo_tabela = body.get("tipo_tabela", "notas_fiscais")
        records = body.get("records") or []
        silver_path = medallion_service.append_to_silver(tipo_tabela, records)
        return {
            "etapa": "2_silver",
            "status": "SUCESSO",
            "tipo_tabela": tipo_tabela,
            "arquivo_silver": os.path.basename(silver_path),
            "caminho_completo": silver_path,
            "total_registros_enviados": len(records),
            "formato": "Apache Parquet (compressão snappy)"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na Etapa 2 (Silver): {str(e)}")


@app.post("/api/pipeline/gold")
async def step_gold(request: Request):
    """
    PASSO 3 DO N8N: Promoção para Gold, DuckDB Analítico e dbt-duckdb
    Promove o .parquet consolidado de Silver para Gold, sincroniza a tabela no DuckDB
    e atualiza a tabela catalogo_metadados via dbt.
    """
    try:
        body = await request.json()
        tipo_tabela = body.get("tipo_tabela", "notas_fiscais")
        gold_path = medallion_service.promote_to_gold(tipo_tabela)
        
        # Recupera as colunas registradas no catálogo para demonstrar ao aluno
        catalogo_amostra = []
        try:
            with db.get_connection() as con:
                rows = con.execute("""
                    SELECT tabela, campo, tipo_dado, camada_origem 
                    FROM catalogo_metadados 
                    WHERE tabela = ?
                    ORDER BY posicao_ordem
                """, [tipo_tabela]).fetchall()
                catalogo_amostra = [{"tabela": r[0], "campo": r[1], "tipo": r[2], "camada": r[3]} for r in rows]
        except Exception:
            pass

        return {
            "etapa": "3_gold_dbt",
            "status": "SUCESSO",
            "tipo_tabela": tipo_tabela,
            "arquivo_gold": os.path.basename(gold_path),
            "caminho_completo": gold_path,
            "tabela_analitica_duckdb": f"main.{tipo_tabela}",
            "catalogo_metadados_atualizado": True,
            "colunas_catalogadas": catalogo_amostra
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na Etapa 3 (Gold/dbt): {str(e)}")


@app.post("/api/pipeline/vectorize")
async def step_vectorize(request: Request):
    """
    PASSO 4 DO N8N: Registro Relacional, Chunking, Tokenização, FastEmbed e Vetorização DuckDB
    Conclui o pipeline registrando o documento relacional em main.documentos
    e seus chunks vetorizados em main.documento_chunks.
    """
    try:
        body = await request.json()
        nome_arquivo = body.get("nome_arquivo")
        categoria = body.get("categoria", "Notas Fiscais")
        secao = body.get("secao", "FINANCEIRO")
        parsed = body.get("campos_extraidos") or {}

        doc_payload = {
            "nome_arquivo": nome_arquivo,
            "categoria": categoria,
            "total_campos": parsed.get("total_campos", 0),
            "campos_preenchidos": parsed.get("campos_preenchidos", 0),
            "campos_nulos": parsed.get("campos_nulos", 0),
            "percentual_sucesso": parsed.get("percentual_sucesso", 100.0),
            "status": parsed.get("status", "PROCESSADO")
        }
        saved_doc = db.insert_documento(doc_payload)
        doc_id = saved_doc.get("id")

        # 1. Chunking
        raw_chunks = chunker.chunk_invoice(doc_id, parsed)

        # 2. Tokenização Didática
        tokenized_chunks = []
        texts_to_embed = []
        for chk in raw_chunks:
            tok_info = tokenizer.tokenize_text(chk["texto"])
            tokenized_chunks.append({
                "posicao": chk["posicao_chunk"],
                "tipo": chk["tipo_secao"],
                "texto": chk["texto"],
                "token_stats": tok_info,
                "metadados": chk["metadados"]
            })
            texts_to_embed.append(chk["texto"])

        # 3. Embeddings FastEmbed
        vectors = embedder.embed_texts(texts_to_embed)

        # 4. Gravação no DuckDB
        db_chunks = []
        chunks_preview = []
        for chk, vec, tok_item in zip(raw_chunks, vectors, tokenized_chunks):
            enhanced_metadata = {
                **chk["metadados"],
                "nome_arquivo": nome_arquivo,
                "secao": secao,
                "total_tokens": tok_item["token_stats"]["total_tokens"],
                "chars": tok_item["token_stats"]["char_count"],
                "ratio_chars_token": tok_item["token_stats"]["ratio_chars_per_token"]
            }
            db_chunks.append({
                "documento_id": doc_id,
                "texto": chk["texto"],
                "vetor": vec,
                "metadados": enhanced_metadata
            })
            chunks_preview.append({
                "posicao": chk["posicao_chunk"],
                "tipo": chk["tipo_secao"],
                "tokens": tok_item["token_stats"]["total_tokens"],
                "vetor_preview": embedder.get_embedding_preview(vec)
            })

        db.insert_chunks(db_chunks)

        return {
            "etapa": "4_vetorizacao_rag",
            "status": "SUCESSO",
            "documento_id": doc_id,
            "nome_arquivo": nome_arquivo,
            "tabela_relacional": "main.documentos",
            "tabela_vetorial": "main.documento_chunks",
            "total_chunks_gerados": len(db_chunks),
            "chunks_detalhes": chunks_preview
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na Etapa 4 (Vetorização): {str(e)}")

