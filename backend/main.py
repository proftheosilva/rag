import os
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
    Varre as subpastas da pasta raw (ex: FINANCEIRO) listando os arquivos disponíveis
    e compara com o resumo de indexação no DuckDB.
    """
    try:
        summary = db.get_ingestion_summary()
    except Exception:
        raise HTTPException(status_code=503, detail="Não foi possível consultar a ingestão no DuckDB.")

    scanned = rag_service.scan_raw_files(subfolder=secao)
    local_files = [item["relative_path"] for item in scanned]
    filenames = [item["filename"] for item in scanned]

    return {
        "files": sorted(local_files),
        "filenames": sorted(filenames),
        "detalhes": scanned,
        "secao_filtro": secao,
        "total_arquivos_locais": len(local_files),
        "total_documentos_vetorizados": summary["total_documentos_vetorizados"],
        "total_chunks_vetorizados": summary["total_chunks_vetorizados"],
    }

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
