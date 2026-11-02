import os
import json
import uuid
import datetime
import logging
from typing import List, Dict, Any, Optional
import duckdb
from backend.config import settings

logger = logging.getLogger(__name__)

class DuckDBManager:
    """
    Gerenciador do banco analítico e vetorial DuckDB.
    Substitui o Supabase armazenando tanto dados relacionais (documentos)
    quanto dados vetoriais com similaridade de cosseno (documento_chunks)
    e tabelas analíticas derivadas da camada Gold.
    """
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.DUCKDB_PATH
        # Garante que o diretório database exista
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def get_connection(self, read_only: bool = False):
        """Retorna uma nova conexão ao DuckDB com tolerância a locks transitórios."""
        import time
        for attempt in range(5):
            try:
                return duckdb.connect(self.db_path, read_only=read_only)
            except Exception as e:
                if attempt == 4:
                    # Fallback para read_only se for erro de lock em consulta
                    try:
                        return duckdb.connect(self.db_path, read_only=True)
                    except Exception:
                        raise e
                time.sleep(0.4 * (attempt + 1))

    def _init_db(self):
        """Inicializa as tabelas relacionais e vetoriais de documentos."""
        with self.get_connection() as con:
            # 1. Tabela relacional de Documentos
            con.execute("""
                CREATE TABLE IF NOT EXISTS documentos (
                    id VARCHAR PRIMARY KEY,
                    nome_arquivo VARCHAR,
                    categoria VARCHAR,
                    total_campos INTEGER,
                    campos_preenchidos INTEGER,
                    campos_nulos INTEGER,
                    percentual_sucesso DOUBLE,
                    status VARCHAR,
                    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Tabela vetorial de Chunks com embeddings
            con.execute("""
                CREATE TABLE IF NOT EXISTS documento_chunks (
                    id VARCHAR PRIMARY KEY,
                    documento_id VARCHAR,
                    texto VARCHAR,
                    vetor FLOAT[],
                    metadados JSON,
                    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (documento_id) REFERENCES documentos(id)
                );
            """)

    def check_connection(self) -> bool:
        try:
            with self.get_connection() as con:
                con.execute("SELECT 1;").fetchone()
            return True
        except Exception as e:
            logger.error(f"Erro ao conectar ao DuckDB: {e}")
            return False

    def insert_documento(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insere um registro na tabela relacional documentos."""
        doc_id = doc_data.get("id") or str(uuid.uuid4())
        criado_em = doc_data.get("criado_em") or datetime.datetime.now().isoformat()
        with self.get_connection() as con:
            con.execute("""
                INSERT INTO documentos (
                    id, nome_arquivo, categoria, total_campos,
                    campos_preenchidos, campos_nulos, percentual_sucesso,
                    status, criado_em
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                doc_id,
                doc_data.get("nome_arquivo", ""),
                doc_data.get("categoria", "Notas Fiscais"),
                doc_data.get("total_campos", 0),
                doc_data.get("campos_preenchidos", 0),
                doc_data.get("campos_nulos", 0),
                doc_data.get("percentual_sucesso", 0.0),
                doc_data.get("status", "PROCESSADO"),
                criado_em
            ])
            res = con.execute("SELECT * FROM documentos WHERE id = ?", [doc_id]).fetchdf()
            if not res.empty:
                return res.to_dict(orient="records")[0]
        return {"id": doc_id, **doc_data}

    def insert_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Insere os chunks com seus respectivos vetores na tabela documento_chunks."""
        if not chunks_data:
            return []
        inserted = []
        with self.get_connection() as con:
            for item in chunks_data:
                chunk_id = item.get("id") or str(uuid.uuid4())
                doc_id = item.get("documento_id")
                texto = item.get("texto", "")
                vetor = [float(v) for v in item.get("vetor", [])]
                meta = json.dumps(item.get("metadados", {}), ensure_ascii=False)
                criado_em = item.get("criado_em") or datetime.datetime.now().isoformat()

                con.execute("""
                    INSERT INTO documento_chunks (
                        id, documento_id, texto, vetor, metadados, criado_em
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, [chunk_id, doc_id, texto, vetor, meta, criado_em])
                inserted.append({"id": chunk_id, "documento_id": doc_id})
        return inserted

    def list_documentos(self) -> List[Dict[str, Any]]:
        """Lista todos os documentos ordenados por data de criação desc."""
        with self.get_connection() as con:
            df = con.execute("SELECT * FROM documentos ORDER BY criado_em DESC").fetchdf()
            if df.empty:
                return []
            # Converte timestamps para strings
            df["criado_em"] = df["criado_em"].astype(str)
            return df.to_dict(orient="records")

    def get_ingestion_summary(self) -> Dict[str, Any]:
        """Retorna resumo dos arquivos indexados e total de chunks."""
        with self.get_connection() as con:
            res = con.execute("""
                SELECT 
                    d.nome_arquivo,
                    count(c.id) as chunk_count
                FROM documentos d
                INNER JOIN documento_chunks c ON d.id = c.documento_id
                GROUP BY d.nome_arquivo
                ORDER BY d.nome_arquivo ASC
            """).fetchall()

            indexed_names = [r[0] for r in res]
            total_chunks = sum(r[1] for r in res)
            return {
                "arquivos_indexados": indexed_names,
                "total_documentos_vetorizados": len(indexed_names),
                "total_chunks_vetorizados": total_chunks,
            }

    def delete_documento(self, doc_id: str):
        """Remove documento e chunks correspondentes."""
        with self.get_connection() as con:
            con.execute("DELETE FROM documento_chunks WHERE documento_id = ?", [doc_id])
            con.execute("DELETE FROM documentos WHERE id = ?", [doc_id])

    def clear_all(self):
        """Limpa todos os documentos e chunks."""
        with self.get_connection() as con:
            con.execute("DELETE FROM documento_chunks;")
            con.execute("DELETE FROM documentos;")

    def match_chunks(
        self,
        query_embedding: List[float],
        match_count: int = 5,
        match_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Executa busca por similaridade de cosseno vetorial nativa no DuckDB.
        Substitui o RPC match_documento_chunks do Supabase / pgvector.
        """
        query_vec = [float(v) for v in query_embedding]
        with self.get_connection() as con:
            query = """
                SELECT 
                    c.id,
                    c.documento_id,
                    c.texto,
                    c.metadados,
                    list_cosine_similarity(c.vetor, ?::FLOAT[]) AS similarity
                FROM documento_chunks c
                WHERE list_cosine_similarity(c.vetor, ?::FLOAT[]) >= ?
                ORDER BY similarity DESC
                LIMIT ?
            """
            rows = con.execute(query, [query_vec, query_vec, float(match_threshold), int(match_count)]).fetchall()
            
            results = []
            for r in rows:
                meta = r[3]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        pass
                results.append({
                    "id": r[0],
                    "documento_id": r[1],
                    "texto": r[2],
                    "metadados": meta,
                    "similarity": float(r[4])
                })
            return results

    def sync_gold_parquet_to_table(self, parquet_path: str, table_name: str):
        """
        Sincroniza um arquivo .parquet da camada Gold para uma tabela analítica no DuckDB.
        Ex: sync_gold_parquet_to_table('.../gold/notas_fiscais.parquet', 'notas_fiscais')
        """
        if not os.path.exists(parquet_path):
            return
        clean_path = parquet_path.replace("\\", "/")
        with self.get_connection() as con:
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{clean_path}');")

db = DuckDBManager()
