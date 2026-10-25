"use strict";

(() => {
  const thread = document.getElementById("thread");
  const empty = document.getElementById("empty");
  const form = document.getElementById("composer");
  const input = document.getElementById("question");
  const sendButton = document.getElementById("send");
  const statusBox = document.getElementById("status");
  const statusText = document.getElementById("status-text");
  const panelTitle = document.getElementById("panel-title");
  const panelSub = document.getElementById("panel-sub");
  const panelList = document.getElementById("panel-list");

  let busy = false;
  let documents = [];

  // ---------- helpers ----------

  const escapeHtml = (text) =>
    String(text).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);

  const el = (tag, className, html) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (html !== undefined) node.innerHTML = html;
    return node;
  };

  const formatSeconds = (ms) => `${(ms / 1000).toFixed(1)} s`;
  const formatCost = (usd) =>
    usd === null || usd === undefined ? "cost n/a" : `≈ $${usd < 0.01 ? usd.toFixed(5) : usd.toFixed(3)}`;

  // Minimal, safe Markdown: escape first, then paragraphs, lists, bold and [n] markers.
  function renderAnswer(text, citedNumbers) {
    const inline = (line) =>
      escapeHtml(line)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\[(\d+)\]/g, (match, n) =>
          citedNumbers.has(Number(n))
            ? `<button type="button" class="cite" data-n="${n}" aria-label="Show source ${n}">${n}</button>`
            : "");

    const blocks = text.trim().split(/\n\s*\n/);
    return blocks.map((block) => {
      const lines = block.split("\n");
      if (lines.every((l) => /^\s*[-*•]\s+/.test(l))) {
        return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*•]\s+/, ""))}</li>`).join("")}</ul>`;
      }
      if (lines.every((l) => /^\s*\d+[.)]\s+/.test(l))) {
        return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)]\s+/, ""))}</li>`).join("")}</ol>`;
      }
      return `<p>${lines.map(inline).join("<br>")}</p>`;
    }).join("");
  }

  const scrollToBottom = () => { thread.scrollTop = thread.scrollHeight; };

  // ---------- right panel ----------

  function showKnowledgeBase() {
    panelTitle.textContent = "Knowledge base";
    panelSub.textContent = `${documents.length} documents indexed for retrieval`;
    panelList.innerHTML = "";
    if (!documents.length) {
      panelList.appendChild(el("li", "panel-empty", "No documents indexed yet. Run the ingestion first."));
      return;
    }
    for (const doc of documents) {
      const item = el("li", "panel-item");
      item.innerHTML = `
        <div class="panel-row">
          <span class="panel-num" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="13" height="13"><path d="M6 3h8l4 4v14H6z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>
          </span>
          <div>
            <div class="panel-title">${escapeHtml(doc.title)}</div>
            <div class="panel-heading">${escapeHtml(doc.path)} · ${doc.chunks} passages</div>
          </div>
        </div>`;
      panelList.appendChild(item);
    }
  }

  function showSources(message, activeNumber) {
    const result = message._result;
    document.querySelectorAll(".msg.assistant.selected").forEach((m) => m.classList.remove("selected"));
    message.classList.add("selected");

    panelTitle.textContent = "Sources";
    const cited = result.sources.filter((s) => s.cited).length;
    if (!result.sources.length) {
      panelSub.textContent = "No passage was relevant enough to answer";
    } else if (result.refused) {
      panelSub.textContent = `${result.sources.length} passages checked · none contains the answer`;
    } else {
      panelSub.textContent = `${result.sources.length} passages retrieved · ${cited} cited in the answer`;
    }
    panelList.innerHTML = "";

    if (!result.sources.length) {
      panelList.appendChild(el("li", "panel-empty",
        "The question did not match any passage above the similarity threshold, so no answer was generated."));
      return;
    }
    for (const source of result.sources) {
      const item = el("li", `panel-item${source.cited ? " cited" : ""}`);
      item.dataset.n = source.number;
      const pct = Math.max(0, Math.min(100, Math.round(source.similarity * 100)));
      item.innerHTML = `
        <button type="button" aria-expanded="false">
          <div class="panel-row">
            <span class="panel-num">${source.number}</span>
            <div>
              <div class="panel-title">${escapeHtml(source.doc_title)}</div>
              <div class="panel-heading">${escapeHtml(source.heading)}</div>
              <div class="panel-tags">
                <span class="bar" title="Cosine similarity"><i></i></span>
                <span>similarity ${source.similarity.toFixed(2)}</span>
                ${source.cited ? '<span class="tag">cited</span>' : '<span class="tag muted">context</span>'}
              </div>
            </div>
          </div>
        </button>
        <div class="panel-text" hidden>${escapeHtml(source.text)}</div>`;
      item.querySelector(".bar > i").style.width = `${pct}%`;
      const button = item.querySelector("button");
      button.addEventListener("click", () => {
        const text = item.querySelector(".panel-text");
        text.hidden = !text.hidden;
        button.setAttribute("aria-expanded", String(!text.hidden));
      });
      panelList.appendChild(item);
    }
    if (activeNumber) highlightPanel(activeNumber, true);
  }

  function highlightPanel(number, expand) {
    panelList.querySelectorAll(".panel-item").forEach((item) => {
      const isActive = Number(item.dataset.n) === number;
      item.classList.toggle("active", isActive);
      if (isActive) {
        if (expand) {
          item.querySelector(".panel-text").hidden = false;
          item.querySelector("button").setAttribute("aria-expanded", "true");
        }
        item.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    });
  }

  // ---------- messages ----------

  function addUserMessage(text) {
    empty.hidden = true;
    const msg = el("div", "msg user");
    msg.appendChild(el("div", "bubble", escapeHtml(text)));
    thread.appendChild(msg);
    scrollToBottom();
  }

  function addPending() {
    const msg = el("div", "msg assistant");
    msg.innerHTML = '<div class="card"><span class="typing" aria-label="Searching the documents"><i></i><i></i><i></i></span></div>';
    thread.appendChild(msg);
    scrollToBottom();
    return msg;
  }

  function toggleSnippet(msg, number) {
    const snippet = msg.querySelector(".snippet");
    const citation = msg._result.citations.find((c) => c.number === number);
    const alreadyOpen = !snippet.hidden && Number(snippet.dataset.n) === number;

    msg.querySelectorAll(".cite, .chip").forEach((b) =>
      b.classList.toggle("active", !alreadyOpen && Number(b.dataset.n) === number));

    if (alreadyOpen || !citation) {
      snippet.hidden = true;
      msg.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-expanded", "false"));
      return;
    }
    snippet.dataset.n = String(number);
    snippet.innerHTML = `
      <div class="snippet-head">
        <div class="snippet-title">[${number}] ${escapeHtml(citation.doc_title)} <span>› ${escapeHtml(citation.heading)}</span></div>
        <div class="snippet-score">similarity ${citation.similarity.toFixed(2)}</div>
      </div>
      <div class="snippet-text">${escapeHtml(citation.snippet)}</div>
      <div class="snippet-file">Source file: <code>${escapeHtml(citation.doc_path)}</code></div>`;
    snippet.hidden = false;
    msg.querySelectorAll(".chip").forEach((c) =>
      c.setAttribute("aria-expanded", String(Number(c.dataset.n) === number)));
    showSources(msg, number);
  }

  function renderResult(msg, result) {
    msg._result = result;
    msg.classList.toggle("refused", result.refused);
    const cited = new Set(result.citations.map((c) => c.number));
    const card = el("div", "card");

    if (result.refused) {
      card.appendChild(el("div", "refusal-label",
        '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 7.5v5.5M12 16.5v.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>Not covered by the documents'));
    }
    card.appendChild(el("div", "answer", renderAnswer(result.answer, cited)));

    if (result.citations.length) {
      const chips = el("div", "chips");
      for (const c of result.citations) {
        const chip = el("button", "chip");
        chip.type = "button";
        chip.dataset.n = c.number;
        chip.setAttribute("aria-expanded", "false");
        chip.innerHTML = `<span class="num">${c.number}</span><span class="label"><b>${escapeHtml(c.doc_title)}</b> › ${escapeHtml(c.heading)}</span>`;
        chips.appendChild(chip);
      }
      card.appendChild(chips);
      const snippet = el("div", "snippet");
      snippet.hidden = true;
      card.appendChild(snippet);
    }

    msg.innerHTML = "";
    msg.appendChild(card);

    const meta = el("div", "meta");
    const tokens = result.usage.input_tokens + result.usage.output_tokens;
    const parts = [formatSeconds(result.latency_ms.total)];
    if (result.model) parts.push(`${tokens.toLocaleString("en-US")} tokens`);
    parts.push(formatCost(result.usage.cost_usd));
    parts.push(result.model ? result.model : "no model call (retrieval gate)");
    meta.innerHTML = parts.map((p) => `<span>${escapeHtml(p)}</span>`).join("");
    msg.appendChild(meta);

    card.addEventListener("click", (event) => {
      const target = event.target.closest(".cite, .chip");
      if (target) toggleSnippet(msg, Number(target.dataset.n));
      else showSources(msg);
    });
    showSources(msg);
    scrollToBottom();
  }

  async function ask(question) {
    if (busy) return;
    busy = true;
    sendButton.disabled = true;
    addUserMessage(question);
    const pending = addPending();
    try {
      const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`;
        throw new Error(detail);
      }
      renderResult(pending, data);
    } catch (error) {
      pending.innerHTML = `<div class="card"><span class="error-text">${escapeHtml(error.message || "Something went wrong.")}</span></div>`;
    } finally {
      busy = false;
      sendButton.disabled = false;
      input.focus();
    }
  }

  // ---------- wiring ----------

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    input.style.height = "";
    ask(question);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });

  document.querySelectorAll(".suggestion").forEach((button) =>
    button.addEventListener("click", () => ask(button.textContent.trim())));

  async function loadStatus() {
    try {
      const [health, docs] = await Promise.all([
        fetch("/health").then((r) => r.json()),
        fetch("/documents").then((r) => r.json()),
      ]);
      documents = docs;
      statusBox.classList.add("ok");
      statusText.textContent = `${health.documents} documents · ${health.chunks} passages · ${health.llm_model}`;
      showKnowledgeBase();
    } catch {
      statusBox.classList.add("error");
      statusText.textContent = "API unreachable";
    }
  }

  loadStatus();
  input.focus();
})();
