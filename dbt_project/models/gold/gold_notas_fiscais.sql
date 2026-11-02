-- Modelo Gold: Notas Fiscais Consolidadas
-- Lê diretamente do Parquet promovido para Gold e gera a tabela analítica oficial com métricas

WITH source_gold AS (
    SELECT * FROM read_parquet('{{ env_var("GOLD_DIR", "../dados/documentos/gold") }}/notas_fiscais.parquet')
)

SELECT
    nome_arquivo,
    numero_nota,
    serie,
    data_emissao,
    fornecedor,
    cnpj_fornecedor,
    cliente,
    cnpj_cliente,
    natureza_operacao,
    COALESCE(valor_total, 0.0) AS valor_total,
    observacoes,
    secao,
    status,
    CURRENT_TIMESTAMP AS dbt_processado_em
FROM source_gold
