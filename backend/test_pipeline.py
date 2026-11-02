import sys
from backend.rag_service import rag_service
from backend.database import db

print("=== 1. Limpando base anterior no DuckDB ===")
try:
    db.clear_all()
    print("DuckDB limpo com sucesso.")
except Exception as e:
    print(f"Erro ao limpar: {e}")

print("\n=== 2. Listando arquivos disponíveis na subpasta FINANCEIRO da camada RAW ===")
scanned = rag_service.scan_raw_files("FINANCEIRO")
print(f"Arquivos encontrados: {[f['filename'] for f in scanned]}")

print("\n=== 3. Ingerindo documentos do FINANCEIRO sob Arquitetura Medalhão + DuckDB ===")
for file_info in scanned:
    fn = file_info["filename"]
    res = rag_service.process_and_ingest_document(fn, source="raw", subfolder="FINANCEIRO")
    print(f" -> Processado {fn}:")
    print(f"    Status: {res['status']} | Tabela analítica: {res['tabela_analitica']}")
    print(f"    Silver: {res['silver_parquet']} | Gold: {res['gold_parquet']}")
    print(f"    Total chunks vetoriais: {res['total_chunks']}")

print("\n=== 4. Verificando Tabelas Analíticas e Chunks no DuckDB ===")
with db.get_connection() as con:
    tables = con.execute("SHOW TABLES;").fetchall()
    print(f"Tabelas criadas no DuckDB: {[t[0] for t in tables]}")
    count_docs = con.execute("SELECT count(*) FROM documentos;").fetchone()[0]
    count_chunks = con.execute("SELECT count(*) FROM documento_chunks;").fetchone()[0]
    count_nf = con.execute("SELECT count(*) FROM notas_fiscais;").fetchone()[0]
    print(f"Registros -> documentos: {count_docs} | documento_chunks: {count_chunks} | notas_fiscais: {count_nf}")

print("\n=== 5. Testando Consulta RAG COM dados existentes no DuckDB ===")
q1 = "Qual é o serviço prestado e a empresa emitente da nota fiscal 8010?"
res_q1 = rag_service.query_rag(q1, top_k=3)
print(f"Pergunta: {q1}")
print(f"Status: {res_q1['status']}")
print(f"Total candidatos recuperados no DuckDB: {len(res_q1['candidatos'])}")
for c in res_q1['candidatos'][:3]:
    print(f"  #{c['ranking']} - Doc: {c['nome_arquivo']} - Score {c['score_similaridade']} - Seção: {c['tipo_secao']}")
print(f"Resposta LLM:\n{res_q1['resposta']}")

print("\n=== 6. Testando Consulta RAG SEM dados (Regra de Ausência de Dados) ===")
q2 = "Qual o resultado do jogo de futebol de ontem à noite?"
res_q2 = rag_service.query_rag(q2, top_k=5, min_threshold=0.55)
print(f"Pergunta: {q2}")
print(f"Status: {res_q2['status']}")
print(f"LLM Executado: {res_q2['llm_executado']}")
print(f"Resposta exata: '{res_q2['resposta']}'")
assert res_q2['resposta'] == "Sem dados disponiveis na base", "Falha na regra de ausência de dados!"

print("\n=== TODOS OS TESTES MEDALHÃO + DUCKDB PASSARAM COM SUCESSO! ===")
