"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
const labels = {
  TOKENIZACAO: "Tokenização didática",
  EMBEDDING: "Embedding da pergunta",
  RETRIEVAL: "Recuperação vetorial",
  PROMPT_AUMENTADO: "Prompt aumentado",
  LLM_OPENROUTER: "Geração pela LLM",
};
async function api(path, body) {
  const res = await fetch(
    `/api/${path}`,
    body
      ? {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {},
  );
  const data = await res.json();
  if (!res.ok)
    throw Error(data.detail || "Falha na comunicação com o backend.");
  return data;
}
function Trace({ data }) {
  return (
    <details className="trace">
      <summary>Inspecionar pipeline RAG · {data.tempo_total_ms} ms</summary>
      <p className="notice">
        Os tempos abaixo são medidos pelo servidor. Esta linha do tempo é
        exibida após a resposta, sem simular progresso em tempo real.
      </p>
      <div className="timeline">
        {Object.entries(labels).map(([key, label], i) => {
          const step = data.pipeline_steps?.find((s) => s.etapa === key);
          return (
            <article className={step ? "step" : "step skipped"} key={key}>
              <span className="number">{i + 1}</span>
              <div>
                <h4>
                  {label}{" "}
                  <small>
                    {step ? `${step.tempo_ms} ms` : "Não executado"}
                  </small>
                </h4>
                {step && (
                  <>
                    <p>{step.descricao}</p>
                    {key === "TOKENIZACAO" && (
                      <>
                        <p className="notice">
                          tiktoken cl100k_base: demonstração independente. Estes
                          IDs não são os tokens internos do FastEmbed nem
                          necessariamente os da LLM.
                        </p>
                        <div className="tokens">
                          {step.detalhes.preview_tokens.map((t, j) => (
                            <span key={j}>
                              <code>{t.token}</code>
                              <small>ID {t.id}</small>
                            </span>
                          ))}
                        </div>
                        <p>
                          {step.detalhes.total_tokens} tokens · amostra exibida:{" "}
                          {step.detalhes.preview_tokens.length}
                        </p>
                      </>
                    )}
                    {key === "EMBEDDING" && (
                      <>
                        <p>
                          {step.detalhes.dimension} dimensões · norma L2{" "}
                          {step.detalhes.norm_l2}
                        </p>
                        <pre>
                          {JSON.stringify(step.detalhes.first_5_values)}
                        </pre>
                        <p className="notice">
                          Amostra das primeiras coordenadas do vetor. O score de
                          cosseno não é uma probabilidade de acerto.
                        </p>
                      </>
                    )}
                    {key === "RETRIEVAL" && (
                      <p>
                        {step.total_recuperados} candidatos recuperados · Top-K{" "}
                        {step.top_k_solicitado}
                      </p>
                    )}
                    {key === "PROMPT_AUMENTADO" && (
                      <pre>{data.prompt_aumentado}</pre>
                    )}
                    {key === "LLM_OPENROUTER" && (
                      <p>
                        Modelo efetivo: {data.modelo_usado || "Não informado"}
                      </p>
                    )}
                  </>
                )}
              </div>
            </article>
          );
        })}
      </div>
    </details>
  );
}
function Evidence({ data }) {
  return (
    <details className="evidence" open>
      <summary>Trechos recuperados · {data.candidatos?.length || 0}</summary>
      <div className="cards">
        {data.candidatos?.map((c) => (
          <article className="chunk" key={c.id}>
            <div className="chunkhead">
              <strong>
                #{c.ranking} · {c.nome_arquivo}
              </strong>
              <span className="badge">
                Score {c.score_similaridade.toFixed(4)}
              </span>
            </div>
            <p className="muted">
              ID {c.id} · Documento {c.documento_id} · {c.tipo_secao}
            </p>
            <span className="tag">
              {data.candidatos_relevantes?.some((r) => r.id === c.id)
                ? "Incluído no contexto"
                : "Abaixo do limiar"}
            </span>
            <p className="excerpt">{c.trecho}</p>
            <details>
              <summary>Metadados</summary>
              <pre>{JSON.stringify(c.metadados, null, 2)}</pre>
            </details>
          </article>
        ))}
      </div>
      {!data.candidatos?.length && <p>Nenhum trecho recuperado.</p>}
    </details>
  );
}
export default function Page() {
  const [theme, setTheme] = useState("dark");
  useEffect(() => {
    try {
      const saved = localStorage.getItem("rag-theme");
      if (saved === "light" || saved === "dark") {
        setTheme(saved);
        document.documentElement.dataset.theme = saved;
      }
    } catch { /* O tema continua disponível se o armazenamento estiver bloqueado. */ }
  }, []);
  function changeTheme(value) {
    setTheme(value);
    document.documentElement.dataset.theme = value;
    try { localStorage.setItem("rag-theme", value); } catch {}
  }

  const [tab, setTab] = useState("chat"),
    [messages, setMessages] = useState([]),
    [question, setQuestion] = useState(""),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [status, setStatus] = useState(null),
    [files, setFiles] = useState([]),
    [indexSummary, setIndexSummary] = useState({
      total_arquivos_locais: 0,
      total_documentos_vetorizados: 0,
      total_chunks_vetorizados: 0,
    }),
    [filesLoading, setFilesLoading] = useState(true),
    [ingestion, setIngestion] = useState(null),
    [k, setK] = useState(5),
    [threshold, setThreshold] = useState(0.35);
  const end = useRef(null);
  const filesRequestActive = useRef(false);

  // Atualiza a lista pendente e os totais consultados no Supabase.
  const refreshAvailableFiles = useCallback(async () => {
    // Evita acumular requisições se o servidor estiver respondendo devagar.
    if (filesRequestActive.current) return;
    filesRequestActive.current = true;
    setFilesLoading(true);
    try {
      const data = await api("available-files");
      setFiles(data.files || []);
      setIndexSummary({
        total_arquivos_locais: data.total_arquivos_locais || 0,
        total_documentos_vetorizados: data.total_documentos_vetorizados || 0,
        total_chunks_vetorizados: data.total_chunks_vetorizados || 0,
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setFilesLoading(false);
      filesRequestActive.current = false;
    }
  }, []);

  useEffect(() => {
    api("status")
      .then(setStatus)
      .catch(() => setStatus({ status: "offline" }));
    refreshAvailableFiles();
  }, [refreshAvailableFiles]);

  // Enquanto a aba de ingestão está aberta, detecta PDFs novos sem recarregar a página.
  useEffect(() => {
    if (tab !== "ingest") return;

    const refreshWhenVisible = () => {
      if (document.visibilityState === "visible") refreshAvailableFiles();
    };
    const timer = window.setInterval(refreshWhenVisible, 5000);
    document.addEventListener("visibilitychange", refreshWhenVisible);

    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", refreshWhenVisible);
    };
  }, [tab, refreshAvailableFiles]);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);
  async function send(e) {
    e.preventDefault();
    if (!question.trim() || busy) return;
    const text = question.trim();
    setQuestion("");
    setError("");
    setBusy(true);
    setMessages((m) => [...m, { role: "user", text }]);
    try {
      const data = await api("query", {
        pergunta: text,
        top_k: Number(k),
        min_threshold: Number(threshold),
      });
      setMessages((m) => [...m, { role: "assistant", data }]);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }
  async function ingest(filename) {
    setBusy(true);
    setError("");
    try {
      const data = await api("ingest", filename ? { filename } : {});
      setIngestion(data);
      await refreshAvailableFiles();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="shell">
      <aside>
        <a className="brand brand-official" href="/" aria-label="Cucando TI — início">
<div className="brand-art"><div className="vinyl-frame"><img src="/brand/vinil.png" alt="Discos de vinil, símbolo da rádio" /></div><div className="logo-frame"><img src="/brand/cucando-ti.png" alt="Cucando TI — Tecnologia em ação" /></div></div><small className="brand-professor">Professor Teophilo Silva</small></a>
        <p className="eyebrow">LABORATÓRIO DE IA GENERATIVA</p>
        <h1>RAG + Teophilo</h1>
        <p className="muted">
          Da pergunta às evidências.
          <br />
          Um pipeline que você pode inspecionar.
        </p>
        <nav>
          <button
            className={tab === "chat" ? "active" : ""}
            onClick={() => setTab("chat")}
          >
            ◈ Consulta RAG
          </button>
          <button
            className={tab === "ingest" ? "active" : ""}
            onClick={() => {
              setTab("ingest");
              refreshAvailableFiles();
            }}
          >
            ▤ Ingestão de documentos
          </button>
        </nav>
        <div className="settings">
          <label>
            Trechos buscados (Top-K)
            <input
              type="number"
              min="1"
              max="20"
              value={k}
              onChange={(e) => setK(e.target.value)}
            />
          </label>
          <label>
            Limiar de similaridade
            <input
              type="number"
              min="0"
              max="1"
              step=".05"
              value={threshold}
              onChange={(e) => setThreshold(e.target.value)}
            />
          </label>
        </div>
        <div className="service">
          <strong>Conexões & Catálogo</strong>
          <p>Backend · {status?.status || "Verificando…"}</p>
          <p>DuckDB · {status?.services?.duckdb || "Não verificado"}</p>
          <p>Catálogo NocoDB · <a href="http://localhost:8085" target="_blank" rel="noreferrer" style={{color: "inherit", textDecoration: "underline"}}>porta 8085 ↗</a></p>
          <p>OpenRouter · {status?.services?.openrouter || "Não verificado"}</p>
        </div>
        <p className="notice">
          As chaves ficam no backend. Cada pergunta é independente: o histórico
          visual não é enviado como memória.
        </p>
      </aside>
      <main>
        <header>
          <label className="theme-picker">
            Tema
            <select aria-label="Selecionar tema" value={theme} onChange={e => changeTheme(e.target.value)}>
              <option value="dark">Escuro</option>
              <option value="light">Claro</option>
            </select>
          </label>

          <div>
            <span className="eyebrow">
              RECUPERAR · CONTEXTUALIZAR · RESPONDER
            </span>
            <h2>
              {tab === "chat"
                ? "Converse com seus documentos"
                : "Prepare sua base de conhecimento"}
            </h2>
          </div>
          {tab === "chat" && (
            <button
              className="secondary"
              disabled={busy}
              onClick={() => {
                setMessages([]);
                setError("");
              }}
            >
              Nova conversa
            </button>
          )}
        </header>
        {error && (
          <div role="alert" className="error">
            {error} <button onClick={() => setError("")}>Fechar</button>
          </div>
        )}
        {tab === "chat" ? (
          <>
            <section className="conversation" aria-live="polite">
              {!messages.length && (
                <div className="welcome">
                  <div className="orb">◈</div>
                  <h2>O que vamos descobrir na base?</h2>
                  <p>
                    Consulte as notas fiscais e veja quais trechos sustentam a
                    resposta.
                  </p>
                  <div className="suggestions">
                    {[
                      "Quais notebooks foram comprados?",
                      "Quais pneus foram adquiridos para a frota?",
                      "Quanto custou a tinta acrílica?",
                    ].map((q) => (
                      <button key={q} onClick={() => setQuestion(q)}>
                        {q} ↗
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={i} className="user">
                    {m.text}
                  </div>
                ) : (
                  <article className="assistant" key={i}>
                    <div className="answerlabel">
                      ◈ RAG + Teophilo{" "}
                      <span className="tag">
                        {m.data.status === "SEM_DADOS"
                          ? "Sem contexto suficiente"
                          : "Resposta gerada"}
                      </span>
                    </div>
                    <div className="markdown">
                      <Markdown remarkPlugins={[remarkGfm]}>
                        {m.data.resposta}
                      </Markdown>
                    </div>
                    <div className="metrics">
                      <span>
                        Modelo:{" "}
                        {m.data.modelo_usado ||
                          (m.data.llm_executado
                            ? "Não informado"
                            : "LLM não acionada")}
                      </span>
                      <span>
                        Entrada: {m.data.uso_tokens?.prompt_tokens ?? "—"}{" "}
                        tokens
                      </span>
                      <span>
                        Saída: {m.data.uso_tokens?.completion_tokens ?? "—"}{" "}
                        tokens
                      </span>
                      <span>
                        Total: {m.data.uso_tokens?.total_tokens ?? "—"}
                      </span>
                      <span>{m.data.tempo_total_ms} ms</span>
                    </div>
                    <Trace data={m.data} />
                    <Evidence data={m.data} />
                  </article>
                ),
              )}
              {busy && (
                <p className="loading">
                  Consultando o backend: recuperação e geração em andamento…
                </p>
              )}
              <div ref={end} />
            </section>
            <form className="composer" onSubmit={send}>
              <textarea
                aria-label="Sua pergunta"
                placeholder="Faça uma pergunta sobre os documentos…"
                value={question}
                disabled={busy}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send(e);
                  }
                }}
              />
              <button disabled={busy || !question.trim()} type="submit">
                {busy ? "Aguarde…" : "Enviar ↑"}
              </button>
              <small>
                Enter envia · Shift + Enter quebra linha · Confira as
                evidências: geração não garante exatidão.
              </small>
            </form>
          </>
        ) : (
          <section className="ingestion">
            <h3>PDFs disponíveis na pasta documentos/raw</h3>
            <p className="notice" aria-live="polite">
              No Supabase: {indexSummary.total_documentos_vetorizados} documentos
              vetorizados · {indexSummary.total_chunks_vetorizados} chunks salvos.
            </p>
            <p>
              Extração de texto → chunking → inspeção de tokens → embeddings →
              Supabase pgvector.
            </p>
            <p className="notice">
              O leitor atual extrai texto de PDFs; não realiza OCR de páginas
              escaneadas. O filtro identifica cada PDF pelo nome do arquivo.
            </p>
            <button disabled={busy || !files.length} onClick={() => ingest()}>
              Ingerir todos os PDFs
            </button>
            {ingestion?.mensagem && (
              <p className="success" role="status" aria-live="polite">
                {ingestion.mensagem}
              </p>
            )}
            <ul>
              {files.map((f) => (
                <li key={f}>
                  {f}
                  <button
                    className="secondary"
                    disabled={busy}
                    onClick={() => ingest(f)}
                  >
                    Ingerir
                  </button>
                </li>
              ))}
            </ul>
            {!files.length && (
              <p>
                {filesLoading
                  ? "Consultando arquivos e índice vetorial…"
                  : indexSummary.total_arquivos_locais
                    ? "Todos os PDFs desta pasta já foram vetorizados."
                    : "Nenhum PDF foi encontrado na pasta documentos/raw."}
              </p>
            )}
            {ingestion?.detalhes?.map((d, i) => (
              <article className="chunk" key={i}>
                <h3>
                  {d.nome_arquivo} · {d.status}
                </h3>
                {d.erro ? (
                  <p className="error">{d.erro}</p>
                ) : (
                  <>
                    <p>
                      {d.total_chunks} chunks ·{" "}
                      {d.metricas_tempo?.tempo_total_ms} ms
                    </p>
                    <details>
                      <summary>Inspecionar dados e etapas da ingestão</summary>
                      <pre>{JSON.stringify(d, null, 2)}</pre>
                    </details>
                  </>
                )}
              </article>
            ))}
          </section>
        )}
      </main>
    </div>
  );
}
