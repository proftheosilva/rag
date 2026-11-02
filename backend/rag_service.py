import io
import os
import shutil
import time
import uuid
from typing import List, Dict, Any, Optional, Union
import httpx
from backend.config import settings
from backend.database import db
from backend.extractor import extractor
from backend.chunker import chunker
from backend.tokenizer import tokenizer
from backend.embedder import embedder
from backend.medallion_service import medallion_service

class RAGService:
    @staticmethod
    def scan_raw_files(subfolder: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Faz a varredura nas subpastas da pasta RAW.
        Por padrão ou fins didáticos, foca em FINANCEIRO, mas varre todas as subpastas.
        Retorna lista com informações de cada arquivo encontrado:
        [{"filename": ..., "relative_path": ..., "full_path": ..., "secao": ..., "tipo": ...}]
        """
        raw_root = settings.RAW_DIR
        found_files = []
        if not os.path.exists(raw_root):
            return found_files

        # Se subfolder for informado, restringe; caso contrário varre subpastas
        target_dir = os.path.join(raw_root, subfolder) if subfolder else raw_root

        for root, dirs, files in os.walk(target_dir):
            rel_dir = os.path.relpath(root, raw_root)
            secao = "GERAL" if rel_dir == "." else rel_dir.split(os.sep)[0]

            for file in files:
                if file.startswith("."):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, raw_root)
                file_type = extractor.identify_file_type(file)
                found_files.append({
                    "filename": file,
                    "relative_path": rel_path.replace("\\", "/"),
                    "full_path": full_path,
                    "secao": secao,
                    "tipo": file_type
                })
        return found_files

    @staticmethod
    def receive_document(
        pdf_filename: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        file_base64: Optional[str] = None,
        file_url: Optional[str] = None,
        from_webhook: Optional[bool] = None,
        source: Optional[str] = None,
        subfolder: Optional[str] = "FINANCEIRO"
    ) -> Dict[str, Any]:
        """
        Lógica centralizada e condicional de recebimento de documentos.
        Suporta:
        1. Pasta RAW local (com suporte a subpastas por seção/divisão ex: FINANCEIRO)
        2. Webhook (binário, base64 ou URL remota)
        """
        is_webhook = False
        if from_webhook is True or (source and source.lower() == "webhook"):
            is_webhook = True
        elif from_webhook is False or (source and source.lower() == "raw"):
            is_webhook = False
        elif file_bytes is not None or file_base64 is not None or file_url is not None:
            is_webhook = True
        else:
            is_webhook = settings.is_webhook_source

        # Webhook
        if is_webhook:
            raw_bytes: Optional[bytes] = None
            if file_bytes is not None:
                raw_bytes = file_bytes
            elif file_base64 is not None:
                import base64
                try:
                    b64_content = file_base64
                    if "," in b64_content:
                        b64_content = b64_content.split(",", 1)[1]
                    raw_bytes = base64.b64decode(b64_content)
                except Exception as e:
                    raise ValueError(f"Falha ao decodificar base64 recebido via webhook: {e}")
            elif file_url is not None:
                try:
                    with httpx.Client(timeout=30.0) as client:
                        resp = client.get(file_url)
                        resp.raise_for_status()
                        raw_bytes = resp.content
                except Exception as e:
                    raise ValueError(f"Falha ao baixar documento do webhook ({file_url}): {e}")
            else:
                raise ValueError("Para recebimento via webhook, é obrigatório fornecer o conteúdo binário, base64 ou URL.")

            clean_filename = os.path.basename(pdf_filename) if pdf_filename else f"webhook_{uuid.uuid4().hex[:8]}.pdf"
            file_type = extractor.identify_file_type(clean_filename)

            return {
                "file_input": io.BytesIO(raw_bytes),
                "filename": clean_filename,
                "file_bytes": raw_bytes,
                "origin": "webhook",
                "secao": subfolder or "WEBHOOK",
                "file_type": file_type
            }

        # Pasta RAW Local
        else:
            if not pdf_filename:
                raise ValueError("Para recebimento via pasta raw, é obrigatório informar o nome ou caminho do arquivo.")

            # Verifica se o arquivo foi informado com caminho relativo ou apenas nome
            target_path = None
            raw_root = settings.RAW_DIR

            # 1. Caminho direto relativo ou absoluto
            if os.path.isabs(pdf_filename) and os.path.exists(pdf_filename):
                target_path = pdf_filename
            else:
                # 2. Busca dentro da subpasta indicada (ex: FINANCEIRO)
                candidate = os.path.join(raw_root, subfolder or "FINANCEIRO", pdf_filename)
                if os.path.exists(candidate):
                    target_path = candidate
                else:
                    # 3. Busca recursiva na pasta raw
                    for root, _, files in os.walk(raw_root):
                        if pdf_filename in files or os.path.basename(pdf_filename) in files:
                            target_path = os.path.join(root, os.path.basename(pdf_filename))
                            break

            if not target_path or not os.path.exists(target_path):
                raise FileNotFoundError(f"Arquivo não encontrado na pasta raw (ou subpastas): {pdf_filename}")

            clean_filename = os.path.basename(target_path)
            rel_dir = os.path.relpath(os.path.dirname(target_path), raw_root)
            secao = "GERAL" if rel_dir == "." else rel_dir.split(os.sep)[0]
            file_type = extractor.identify_file_type(clean_filename)

            with open(target_path, "rb") as f:
                raw_bytes = f.read()

            return {
                "file_input": target_path,
                "filename": clean_filename,
                "file_bytes": raw_bytes,
                "origin": "raw",
                "secao": secao,
                "file_type": file_type
            }

    @staticmethod
    def process_and_ingest_document(
        pdf_filename: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        file_base64: Optional[str] = None,
        file_url: Optional[str] = None,
        from_webhook: Optional[bool] = None,
        source: Optional[str] = None,
        subfolder: Optional[str] = "FINANCEIRO"
    ) -> Dict[str, Any]:
        """
        Executa o pipeline completo na Arquitetura Medalhão + DuckDB:
        1. RAW: Varredura / Recepção do arquivo.
        2. EXTRAÇÃO: Polimórfica (estruturado vs não estruturado).
        3. SILVER: Higienização, remoção de nulos críticos e gravação incremental em Parquet por tipo.
        4. GOLD: Promoção do Parquet consolidado para Gold.
        5. DUCKDB ANALYTICS: Sincronização do Parquet em tabela analítica no DuckDB.
        6. DUCKDB VETORIAL: Criação e gravação relacional (documentos) e vetorial com chunks (documento_chunks).
        """
        start_time = time.time()

        received = RAGService.receive_document(
            pdf_filename=pdf_filename,
            file_bytes=file_bytes,
            file_base64=file_base64,
            file_url=file_url,
            from_webhook=from_webhook,
            source=source,
            subfolder=subfolder
        )

        file_input = received["file_input"]
        filename = received["filename"]
        document_bytes = received["file_bytes"]
        origin = received["origin"]
        secao = received["secao"]
        file_type = received["file_type"]

        # =====================================================================
        # 1. EXTRAÇÃO DEPENDENDO DO TIPO DO DOCUMENTO
        # =====================================================================
        t0 = time.time()
        is_structured = file_type in ["csv", "json", "sql"]

        if is_structured:
            records, categoria, table_type = extractor.extract_structured(file_input, filename, file_type)
            raw_text = "\n".join([str(r) for r in records])
            parsed = {
                "nome_arquivo": filename,
                "categoria": categoria,
                "tipo_tabela": table_type,
                "numero_nota": str(records[0].get("numero_nota", "N/A")) if records else "N/A",
                "data_emissao": str(records[0].get("data_emissao", "")) if records else "",
                "fornecedor": str(records[0].get("fornecedor", "")) if records else "",
                "total_campos": len(records[0]) if records else 0,
                "campos_preenchidos": len(records[0]) if records else 0,
                "campos_nulos": 0,
                "percentual_sucesso": 100.0,
                "status": "PROCESSADO",
                "texto_extraido": raw_text
            }
            records_to_silver = records
        else:
            # Não estruturado (ex: PDF de Nota Fiscal)
            raw_text = extractor.extract_text_from_pdf(file_input)
            parsed = extractor.parse_invoice_fields(raw_text, filename)
            table_type = parsed.get("tipo_tabela", "notas_fiscais")
            # Converte a nota fiscal em um registro estruturado para o parquet
            records_to_silver = [{
                "nome_arquivo": filename,
                "numero_nota": parsed.get("numero_nota"),
                "serie": parsed.get("serie"),
                "data_emissao": parsed.get("data_emissao"),
                "fornecedor": parsed.get("fornecedor"),
                "cnpj_fornecedor": parsed.get("cnpj_fornecedor"),
                "cliente": parsed.get("cliente"),
                "cnpj_cliente": parsed.get("cnpj_cliente"),
                "natureza_operacao": parsed.get("natureza_operacao"),
                "valor_total": parsed.get("valor_total"),
                "observacoes": parsed.get("observacoes"),
                "itens": parsed.get("itens"),
                "secao": secao,
                "status": parsed.get("status")
            }]
        extraction_time_ms = round((time.time() - t0) * 1000, 2)

        # =====================================================================
        # 2. CAMADA SILVER: Tratamento, Remoção de Nulos e Parquet Incremental
        # =====================================================================
        t_silver = time.time()
        silver_path = medallion_service.append_to_silver(table_type, records_to_silver)
        silver_time_ms = round((time.time() - t_silver) * 1000, 2)

        # =====================================================================
        # 3. CAMADA GOLD: Promoção do Parquet e Sincronização DuckDB Analítico
        # =====================================================================
        t_gold = time.time()
        gold_path = medallion_service.promote_to_gold(table_type)
        gold_time_ms = round((time.time() - t_gold) * 1000, 2)

        # =====================================================================
        # 4. GRAVAÇÃO RELACIONAL NO DUCKDB (Tabela 'documentos')
        # =====================================================================
        doc_payload = {
            "nome_arquivo": filename,
            "categoria": parsed["categoria"],
            "total_campos": parsed["total_campos"],
            "campos_preenchidos": parsed["campos_preenchidos"],
            "campos_nulos": parsed["campos_nulos"],
            "percentual_sucesso": parsed["percentual_sucesso"],
            "status": parsed["status"]
        }
        saved_doc = db.insert_documento(doc_payload)
        doc_id = saved_doc.get("id")

        # =====================================================================
        # 5. CHUNKING, TOKENIZAÇÃO E FASTEFMED (Pipeline RAG Vetorial)
        # =====================================================================
        t1 = time.time()
        raw_chunks = chunker.chunk_invoice(doc_id, parsed)
        chunking_time_ms = round((time.time() - t1) * 1000, 2)

        t2 = time.time()
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
        tokenization_time_ms = round((time.time() - t2) * 1000, 2)

        t3 = time.time()
        vectors = embedder.embed_texts(texts_to_embed)
        embedding_time_ms = round((time.time() - t3) * 1000, 2)

        # =====================================================================
        # 6. GRAVAÇÃO VETORIAL NO DUCKDB (Tabela 'documento_chunks')
        # =====================================================================
        t4 = time.time()
        db_chunks = []
        for chk, vec, tok_item in zip(raw_chunks, vectors, tokenized_chunks):
            enhanced_metadata = {
                **chk["metadados"],
                "nome_arquivo": filename,
                "secao": secao,
                "total_tokens": tok_item["token_stats"]["total_tokens"],
                "tipo_tabela": table_type
            }
            db_chunks.append({
                "documento_id": doc_id,
                "texto": chk["texto"],
                "vetor": vec,
                "metadados": enhanced_metadata
            })
        saved_chunks = db.insert_chunks(db_chunks)
        db_time_ms = round((time.time() - t4) * 1000, 2)

        total_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "documento_id": doc_id,
            "nome_arquivo": filename,
            "secao": secao,
            "origem": origin,
            "tipo_documento": file_type,
            "tabela_analitica": table_type,
            "silver_parquet": os.path.basename(silver_path),
            "gold_parquet": os.path.basename(gold_path),
            "status": "CONCLUIDO",
            "dados_estruturados": parsed,
            "total_chunks": len(raw_chunks),
            "chunks_detalhes": [
                {
                    "posicao": tc["posicao"],
                    "tipo": tc["tipo"],
                    "texto": tc["texto"],
                    "tokens": tc["token_stats"]["total_tokens"],
                    "amostra_tokens": tc["token_stats"]["preview_tokens"][:10],
                    "vetor_preview": embedder.get_embedding_preview(v)
                }
                for tc, v in zip(tokenized_chunks, vectors)
            ],
            "metricas_tempo": {
                "extracao_ms": extraction_time_ms,
                "silver_parquet_ms": silver_time_ms,
                "gold_duckdb_ms": gold_time_ms,
                "chunking_ms": chunking_time_ms,
                "tokenizacao_ms": tokenization_time_ms,
                "embedding_ms": embedding_time_ms,
                "duckdb_vetorial_ms": db_time_ms,
                "tempo_total_ms": total_time_ms
            }
        }

    @staticmethod
    def process_and_ingest_from_webhook(
        pdf_filename: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        file_base64: Optional[str] = None,
        file_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Ingestão completa a partir de Webhook."""
        return RAGService.process_and_ingest_document(
            pdf_filename=pdf_filename,
            file_bytes=file_bytes,
            file_base64=file_base64,
            file_url=file_url,
            from_webhook=True,
            source="webhook"
        )

    @staticmethod
    def query_rag(question: str, top_k: int = 5, min_threshold: float = 0.35) -> Dict[str, Any]:
        """Executa a consulta RAG utilizando busca vetorial nativa no DuckDB."""
        start_time = time.time()
        pipeline_steps = []

        # Etapa A: Tokenização
        t0 = time.time()
        query_token_info = tokenizer.tokenize_text(question)
        pipeline_steps.append({
            "etapa": "TOKENIZACAO",
            "descricao": "Tokenização da pergunta do usuário com tiktoken (cl100k_base)",
            "detalhes": query_token_info,
            "tempo_ms": round((time.time() - t0) * 1000, 2)
        })

        # Etapa B: Embedding
        t1 = time.time()
        query_vector = embedder.embed_query(question)
        emb_preview = embedder.get_embedding_preview(query_vector)
        pipeline_steps.append({
            "etapa": "EMBEDDING",
            "descricao": f"Geração do vetor numérico de {embedder.dimension} dimensões via FastEmbed ({embedder.model_name})",
            "detalhes": emb_preview,
            "tempo_ms": round((time.time() - t1) * 1000, 2)
        })

        # Etapa C: Retrieval no DuckDB Vetorial
        t2 = time.time()
        raw_candidates = db.match_chunks(query_vector, match_count=top_k, match_threshold=0.0)
        retrieval_time_ms = round((time.time() - t2) * 1000, 2)

        candidatos = []
        for idx, c in enumerate(raw_candidates, 1):
            score = float(c.get("similarity", 0.0))
            metadata = c.get("metadados") or {}
            candidatos.append({
                "ranking": idx,
                "id": c.get("id"),
                "documento_id": c.get("documento_id"),
                "nome_arquivo": metadata.get("nome_arquivo", "Documento"),
                "numero_nota": metadata.get("numero_nota", "N/A"),
                "score_similaridade": round(score, 4),
                "tipo_secao": metadata.get("tipo", metadata.get("secao", "conteudo")),
                "trecho": c.get("texto", ""),
                "metadados": metadata
            })

        pipeline_steps.append({
            "etapa": "RETRIEVAL",
            "descricao": f"Busca vetorial por similaridade de cosseno no DuckDB (top_k={top_k})",
            "total_recuperados": len(candidatos),
            "top_k_solicitado": top_k,
            "tempo_ms": retrieval_time_ms
        })

        # Filtragem por limiar de relevância
        candidatos_relevantes = [c for c in candidatos if c["score_similaridade"] >= min_threshold]

        if not candidatos_relevantes:
            return {
                "pergunta": question,
                "status": "SEM_DADOS",
                "resposta": "Sem dados disponiveis na base",
                "candidatos": candidatos,
                "candidatos_relevantes": [],
                "prompt_aumentado": None,
                "pipeline_steps": pipeline_steps,
                "llm_executado": False,
                "motivo": "Nenhum candidato relevante recuperado no DuckDB vetorial para a pergunta",
                "tempo_total_ms": round((time.time() - start_time) * 1000, 2)
            }

        # Etapa D: Construção do Prompt Aumentado
        t3 = time.time()
        contexto_textos = []
        for c in candidatos_relevantes:
            doc_ref = f"[Candidato #{c['ranking']} - Doc: {c['nome_arquivo']} - Similaridade: {c['score_similaridade']}]"
            contexto_textos.append(f"{doc_ref}\n{c['trecho']}")
        contexto_consolidado = "\n\n---\n\n".join(contexto_textos)

        instrucoes_sistema = (
            "Você é um assistente RAG especializado em conferência e análise de documentos empresariais e fiscais.\n"
            "DIRETRIZES FUNDAMENTAIS:\n"
            "1. Responda à pergunta do usuário de forma clara, objetiva e precisa, baseando-se EXCLUSIVAMENTE no contexto fornecido abaixo.\n"
            "2. Não invente valores, produtos, datas ou fornecedores.\n"
            "3. Se a informação solicitada pelo usuário NÃO estiver presente no contexto fornecido, responda exatamente: 'Sem dados disponiveis na base'.\n"
            "4. Indique os documentos correspondentes (ex: 'De acordo com o documento/NF nº XXX...')."
        )

        prompt_aumentado = (
            f"=== INSTRUÇÕES DO SISTEMA ===\n{instrucoes_sistema}\n\n"
            f"=== CONTEXTO RECUPERADO DO DUCKDB ===\n{contexto_consolidado}\n\n"
            f"=== PERGUNTA DO USUÁRIO ===\n{question}"
        )

        pipeline_steps.append({
            "etapa": "PROMPT_AUMENTADO",
            "descricao": "Montagem do prompt contextualizado unindo Instruções + Contexto Top-K + Pergunta",
            "tamanho_contexto_chars": len(contexto_consolidado),
            "tempo_ms": round((time.time() - t3) * 1000, 2)
        })

        # Etapa E: Chamada ao LLM via OpenRouter
        t4 = time.time()
        resposta_llm = RAGService._call_openrouter(instrucoes_sistema, contexto_consolidado, question)
        llm_time_ms = round((time.time() - t4) * 1000, 2)

        pipeline_steps.append({
            "etapa": "LLM_OPENROUTER",
            "descricao": f"Geração da resposta final via OpenRouter (modelo: {resposta_llm['modelo']})",
            "tempo_ms": llm_time_ms
        })

        return {
            "pergunta": question,
            "status": "SUCESSO",
            "resposta": resposta_llm["texto"],
            "modelo_usado": resposta_llm["modelo"],
            "uso_tokens": resposta_llm["uso_tokens"],
            "candidatos": candidatos,
            "candidatos_relevantes": candidatos_relevantes,
            "prompt_aumentado": prompt_aumentado,
            "contexto_utilizado": contexto_consolidado,
            "pipeline_steps": pipeline_steps,
            "llm_executado": True,
            "tempo_total_ms": round((time.time() - start_time) * 1000, 2)
        }

    @staticmethod
    def _call_openrouter(system_prompt: str, context: str, user_question: str) -> Dict[str, Any]:
        """Comunicação com a API do OpenRouter."""
        if not settings.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY não configurada no .env")

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "RAG Didatico Teophilo"
        }

        user_content = (
            f"Contexto Recuperado:\n{context}\n\n"
            f"Pergunta: {user_question}"
        )

        payload = {
            "model": settings.OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1
        }

        with httpx.Client(timeout=40.0) as client:
            resp = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return {
                        "texto": (choices[0].get("message", {}).get("content") or "").strip(),
                        "modelo": data.get("model", settings.OPENROUTER_MODEL),
                        "uso_tokens": {key: data.get("usage", {}).get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
                    }
                raise ValueError("OpenRouter não retornou uma resposta.")
            else:
                raise ValueError(f"Erro na chamada OpenRouter (status {resp.status_code})")

rag_service = RAGService()
