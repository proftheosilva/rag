import os
import re
import json
import io
from typing import Dict, Any, List, Union, Tuple
import pandas as pd
import pdfplumber
from pypdf import PdfReader

class DocumentExtractor:
    """
    Extrator polimórfico de documentos estruturados, semiestruturados e não-estruturados.
    Classifica por extensão e conteúdo, tabula os dados e extrai campos essenciais.
    """

    @staticmethod
    def identify_file_type(filename: str) -> str:
        """Identifica a extensão normalizada do arquivo."""
        ext = os.path.splitext(filename)[1].lower()
        if ext in [".csv"]:
            return "csv"
        elif ext in [".json"]:
            return "json"
        elif ext in [".sql"]:
            return "sql"
        elif ext in [".pdf"]:
            return "pdf"
        elif ext in [".txt", ".md"]:
            return "text"
        return "unstructured"

    @staticmethod
    def extract_text_from_pdf(pdf_source: Any) -> str:
        """Extrai texto integral de PDF usando pdfplumber com fallback para pypdf."""
        if isinstance(pdf_source, bytes):
            pdf_source = io.BytesIO(pdf_source)

        full_text = ""
        try:
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            with pdfplumber.open(pdf_source) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        except Exception:
            pass

        if not full_text.strip():
            try:
                if hasattr(pdf_source, "seek"):
                    pdf_source.seek(0)
                reader = PdfReader(pdf_source)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        full_text += t + "\n"
            except Exception:
                pass

        return full_text.strip()

    @staticmethod
    def parse_invoice_fields(text: str, filename: str) -> Dict[str, Any]:
        """Identifica campos estruturados na nota fiscal a partir do texto extraído."""
        data: Dict[str, Any] = {
            "nome_arquivo": filename,
            "categoria": "Notas Fiscais",
            "tipo_tabela": "notas_fiscais",
            "numero_nota": None,
            "serie": None,
            "data_emissao": None,
            "fornecedor": None,
            "cnpj_fornecedor": None,
            "endereco_fornecedor": None,
            "cliente": None,
            "cnpj_cliente": None,
            "endereco_cliente": None,
            "natureza_operacao": None,
            "itens": [],
            "valor_total": None,
            "observacoes": None,
            "texto_extraido": text
        }

        # Número da Nota
        num_match = re.search(r"N[ºo°]\s*(\d+)", text, re.IGNORECASE)
        if num_match:
            data["numero_nota"] = num_match.group(1)

        # Série
        serie_match = re.search(r"SÉRIE\s*(\d+)", text, re.IGNORECASE)
        if serie_match:
            data["serie"] = serie_match.group(1)

        # Data de emissão
        data_match = re.search(r"DATA DE EMISS[ÃA]O:\s*([\d/]+)", text, re.IGNORECASE)
        if data_match:
            data["data_emissao"] = data_match.group(1)

        # Fornecedor / Emitente
        forn_match = re.search(r"EMITENTE\s*/\s*FORNECEDOR:\s*([^\n]+)", text, re.IGNORECASE)
        if forn_match:
            data["fornecedor"] = forn_match.group(1).strip()

        # CNPJs
        cnpjs = re.findall(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b", text)
        if len(cnpjs) >= 1:
            data["cnpj_fornecedor"] = cnpjs[0]
        if len(cnpjs) >= 2:
            data["cnpj_cliente"] = cnpjs[1]

        # Cliente / Destinatário
        cli_match = re.search(r"DESTINAT[ÁA]RIO\s*/\s*CLIENTE:\s*([^\n]+)", text, re.IGNORECASE)
        if cli_match:
            data["cliente"] = cli_match.group(1).strip()

        # Natureza da operação
        nat_match = re.search(r"NATUREZA DA OPERA[ÇC][ÃA]O:\s*([^\n]+)", text, re.IGNORECASE)
        if nat_match:
            data["natureza_operacao"] = nat_match.group(1).strip()

        # Valor Total
        val_match = re.search(r"VALOR TOTAL DA NOTA:\s*R\$\s*([\d\.,]+)", text, re.IGNORECASE)
        if val_match:
            val_str = val_match.group(1).replace(".", "").replace(",", ".")
            try:
                data["valor_total"] = float(val_str)
            except ValueError:
                pass

        # Observações / Informações complementares
        obs_match = re.search(r"INFORMA[ÇC][ÕO]ES COMPLEMENTARES:\s*([^\n]+(?:\n[^\n]+)*?)(?=VALOR TOTAL|$)", text, re.IGNORECASE)
        if obs_match:
            data["observacoes"] = obs_match.group(1).strip()

        # Itens / Produtos
        lines = text.split("\n")
        capturing_items = False
        for line in lines:
            line_str = line.strip()
            if "Descrição do Produto" in line_str or "Descricao do Produto" in line_str:
                capturing_items = True
                continue
            if capturing_items and ("INFORMAÇÕES COMPLEMENTARES" in line_str or "VALOR TOTAL" in line_str):
                capturing_items = False
                break
            if capturing_items and line_str:
                price_matches = re.findall(r"R\$\s*([\d\.,]+)", line_str)
                if price_matches:
                    data["itens"].append(line_str)

        # Cálculo de métricas
        core_fields = [
            "numero_nota", "data_emissao", "fornecedor", "cnpj_fornecedor",
            "cliente", "cnpj_cliente", "natureza_operacao", "valor_total", "observacoes"
        ]
        total_campos = len(core_fields)
        preenchidos = sum(1 for f in core_fields if data.get(f) is not None)
        nulos = total_campos - preenchidos
        percentual = round((preenchidos / total_campos) * 100, 2)

        data["total_campos"] = total_campos
        data["campos_preenchidos"] = preenchidos
        data["campos_nulos"] = nulos
        data["percentual_sucesso"] = percentual
        data["status"] = "PROCESSADO" if percentual >= 60 else "PARCIAL"

        return data

    @staticmethod
    def extract_structured(file_input: Union[str, bytes, io.BytesIO], filename: str, file_type: str) -> Tuple[List[Dict[str, Any]], str, str]:
        """
        Extrai e tabula arquivos estruturados ou semi-estruturados (csv, json, sql).
        Retorna (lista_de_registros_dict, categoria, tipo_tabela).
        """
        records: List[Dict[str, Any]] = []
        raw_content = ""

        # Obter string do conteúdo
        if isinstance(file_input, str):
            with open(file_input, "r", encoding="utf-8", errors="replace") as f:
                raw_content = f.read()
        elif isinstance(file_input, bytes):
            raw_content = file_input.decode("utf-8", errors="replace")
        elif hasattr(file_input, "read"):
            file_input.seek(0)
            content = file_input.read()
            if isinstance(content, bytes):
                raw_content = content.decode("utf-8", errors="replace")
            else:
                raw_content = content

        categoria = "Dados Financeiros"
        tipo_tabela = "financeiro_estruturado"

        if "contrato" in filename.lower():
            categoria = "Contratos"
            tipo_tabela = "contratos"
        elif "nota" in filename.lower() or "nf" in filename.lower():
            categoria = "Notas Fiscais"
            tipo_tabela = "notas_fiscais"

        if file_type == "csv":
            df = pd.read_csv(io.StringIO(raw_content))
            records = df.to_dict(orient="records")
        elif file_type == "json":
            data = json.loads(raw_content)
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = [data]
        elif file_type == "sql":
            # Extração simples de linhas INSERT INTO
            insert_pattern = re.findall(r"INSERT\s+INTO\s+([^\s\(]+)\s*(?:\(([^\)]+)\))?\s*VALUES\s*\(([^\)]+)\);?", raw_content, re.IGNORECASE)
            for tbl, cols, vals in insert_pattern:
                tbl_clean = tbl.strip("`'\" ")
                if tbl_clean:
                    tipo_tabela = tbl_clean.lower()
                val_list = [v.strip().strip("'\"") for v in vals.split(",")]
                if cols:
                    col_list = [c.strip().strip("`'\" ") for c in cols.split(",")]
                    record = dict(zip(col_list, val_list))
                else:
                    record = {f"col_{i}": v for i, v in enumerate(val_list)}
                records.append(record)

        # Adiciona nome do arquivo aos registros estruturados
        for r in records:
            r["nome_arquivo"] = filename

        return records, categoria, tipo_tabela

# Instância padrão mantendo compatibilidade
extractor = DocumentExtractor()
InvoiceExtractor = DocumentExtractor
