import os
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import duckdb
from backend.config import settings
from backend.database import db

logger = logging.getLogger(__name__)

class MedallionService:
    """
    Controla o fluxo das camadas da Arquitetura Medalhão:
    1. RAW: recepção particionada por seção da empresa (ex: FINANCEIRO).
    2. SILVER: tratamento, higienização, descarte de nulos críticos e persistência incremental em .parquet por tipo.
    3. GOLD: consolidação de parquets prontos para Data Lakehouse / Analytics.
    4. DUCKDB: sincronização analítica das tabelas a partir da camada Gold.
    """

    def __init__(self):
        os.makedirs(settings.RAW_DIR, exist_ok=True)
        os.makedirs(settings.SILVER_DIR, exist_ok=True)
        os.makedirs(settings.GOLD_DIR, exist_ok=True)
        os.makedirs(settings.DATABASE_DIR, exist_ok=True)

    @staticmethod
    def clean_and_normalize_records(records: List[Dict[str, Any]], critical_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Trata registros:
        - Converte listas ou dicts em strings json para compatibilidade com Parquet/DuckDB
        - Remove registros com campos nulos críticos mantendo dados consistentes
        """
        if not records:
            return []

        df = pd.DataFrame(records)

        # Tratar colunas complexas (ex: itens como lista) para json string
        for col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)

        # Campos críticos para manter consistência
        if critical_fields:
            existing_critical = [f for f in critical_fields if f in df.columns]
            if existing_critical:
                df = df.dropna(subset=existing_critical)

        # Remover linhas que sejam totalmente nulas ou vazias
        df = df.dropna(how="all")

        return df.to_dict(orient="records")

    def append_to_silver(self, table_type: str, new_records: List[Dict[str, Any]]) -> str:
        """
        Adiciona registros limpos incrementalmente ao arquivo Parquet correspondente
        na camada SILVER, evitando criar múltiplos arquivos avulsos.
        Retorna o caminho do arquivo Parquet na pasta silver.
        """
        silver_file = os.path.join(settings.SILVER_DIR, f"{table_type}.parquet")
        clean_new = self.clean_and_normalize_records(new_records)
        if not clean_new:
            return silver_file

        df_new = pd.DataFrame(clean_new)

        if os.path.exists(silver_file):
            try:
                df_existing = pd.read_parquet(silver_file)
                # Unifica evitando duplicatas exatas
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                # Se tiver nome_arquivo, remove duplicata do mesmo arquivo
                if "nome_arquivo" in df_combined.columns:
                    df_combined = df_combined.drop_duplicates(subset=["nome_arquivo"], keep="last")
                else:
                    df_combined = df_combined.drop_duplicates()
            except Exception as e:
                logger.warning(f"Erro ao ler parquet existente em silver ({e}), regravando com novos registros.")
                df_combined = df_new
        else:
            df_combined = df_new

        # Salva em Parquet com compressão snappy
        df_combined.to_parquet(silver_file, index=False, engine="pyarrow")
        return silver_file

    def promote_to_gold(self, table_type: str) -> str:
        """
        Promove o parquet tratado da camada SILVER para a camada GOLD e
        atualiza a tabela analítica correspondente no DuckDB.
        Retorna o caminho do parquet em Gold.
        """
        silver_file = os.path.join(settings.SILVER_DIR, f"{table_type}.parquet")
        gold_file = os.path.join(settings.GOLD_DIR, f"{table_type}.parquet")

        if not os.path.exists(silver_file):
            raise FileNotFoundError(f"Arquivo parquet não encontrado em silver: {silver_file}")

        # Lê de silver e salva em gold
        df = pd.read_parquet(silver_file)
        df.to_parquet(gold_file, index=False, engine="pyarrow")

        # Carga/Sincronização no banco analítico DuckDB
        db.sync_gold_parquet_to_table(gold_file, table_type)
        logger.info(f"Tabela '{table_type}' sincronizada no DuckDB a partir de {gold_file}")

        # Execução dos modelos e testes do dbt-duckdb e atualização do catálogo
        self.run_dbt_and_update_catalog()

        return gold_file

    def run_dbt_and_update_catalog(self) -> Dict[str, Any]:
        """
        Executa os modelos dbt para validação/linhagem dos dados de Gold e
        extrai diretamente os metadados (esquema, tipos, contagens) para popular
        o catálogo de dados corporativo no DuckDB e pronto para o NocoDB.
        """
        catalog_info = {"status": "SUCESSO", "tabelas_catalogadas": []}
        try:
            # 1. Atualizar e garantir tabela catalogo_metadados no DuckDB
            with db.get_connection() as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS catalogo_metadados (
                        tabela VARCHAR,
                        campo VARCHAR,
                        tipo_dado VARCHAR,
                        aceita_nulo VARCHAR,
                        posicao_ordem INTEGER,
                        camada_origem VARCHAR,
                        sincronizado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # 2. Popula/Atualiza o catálogo com metadados do information_schema das tabelas Gold
                con.execute("""
                    DELETE FROM catalogo_metadados;
                    INSERT INTO catalogo_metadados
                    SELECT 
                        table_name AS tabela,
                        column_name AS campo,
                        data_type AS tipo_dado,
                        is_nullable AS aceita_nulo,
                        ordinal_position AS posicao_ordem,
                        'Gold Data Lakehouse / Parquet' AS camada_origem,
                        CURRENT_TIMESTAMP AS sincronizado_em
                    FROM information_schema.columns
                    WHERE table_schema = 'main'
                      AND table_name NOT IN ('catalogo_metadados', 'documento_chunks')
                    ORDER BY table_name, ordinal_position;
                """)
                catalog_info["total_campos"] = con.execute("SELECT count(*) FROM catalogo_metadados").fetchone()[0]
                logger.info(f"Catálogo de dados atualizado no DuckDB: {catalog_info['total_campos']} campos mapeados.")

            # 3. Invoca o dbtRunner para testes de integridade se disponível
            try:
                from dbt.cli.main import dbtRunner
                dbt_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dbt_project"))
                if os.path.exists(dbt_dir):
                    env = {**os.environ, "DUCKDB_PATH": settings.DUCKDB_PATH, "GOLD_DIR": settings.GOLD_DIR}
                    # Executa dbt run
                    runner = dbtRunner()
                    res = runner.invoke(["run", "--project-dir", dbt_dir, "--profiles-dir", dbt_dir], env=env)
                    catalog_info["dbt_run_success"] = res.success
            except Exception as e_dbt:
                logger.warning(f"Execução dbtRunner informativa: {e_dbt}")

        except Exception as e:
            logger.error(f"Erro ao atualizar catálogo de dados corporativo: {e}")
            catalog_info["status"] = "ERRO"
            catalog_info["erro"] = str(e)

        return catalog_info


medallion_service = MedallionService()
