-- Modelo de Catálogo de Dados Corporativo
-- Extrai campos, tipos e metadados do information_schema das tabelas geradas a partir do Parquet

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
ORDER BY table_name, ordinal_position
