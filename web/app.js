// Minimal UI for the Clinical Trial Protocol QA API (same-origin: /health, /ask).

const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("q");
const sendBtn = document.getElementById("send");

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function addCitations(container, sources) {
  if (!sources || !sources.length) return;
  const d = document.createElement("details");
  d.className = "cites";
  const s = document.createElement("summary");
  s.textContent = `${sources.length} source(s) — hybrid fusion`;
  d.appendChild(s);
  const ul = document.createElement("ul");
  for (const src of sources) {
    const li = document.createElement("li");
    const ranks = src.retriever_ranks ? ` ${JSON.stringify(src.retriever_ranks)}` : "";
    li.textContent = `${src.source} (fusion ${src.fusion_score})${ranks}: ${src.excerpt || ""}`;
    ul.appendChild(li);
  }
  d.appendChild(ul);
  container.appendChild(d);
}

async function ask(question) {
  addMessage(question, "user");
  input.value = "";
  sendBtn.disabled = true;
  const bot = addMessage("…", "bot");
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    bot.textContent = data.answer || "(no answer)";
    addCitations(bot, data.sources);
  } catch (err) {
    bot.textContent = `Error: ${err.message}`;
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (q) ask(q);
});

document.getElementById("examples").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") ask(e.target.textContent);
});

fetch("/health").then((r) => r.json()).then((h) => {
  document.getElementById("backend").textContent = `backend: ${h.backend}`;
  document.getElementById("chunks").textContent = `index: ${h.indexed_chunks} chunks`;
}).catch(() => {});
