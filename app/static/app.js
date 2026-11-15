const $ = (id) => document.getElementById(id);
let busy = false;
function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Please check your input and try again.');
  return data;
}
async function refresh() {
  const docs = await api('/api/documents');
  $('doc-count').textContent = docs.length;
  $('documents').replaceChildren();
  for (const doc of docs) {
    const row = element('div', undefined, 'doc');
    const info = element('div');
    info.append(element('strong', '▤  ' + doc.name), element('small', `${doc.chunks} searchable passages`));
    const remove = element('button', '×');
    remove.setAttribute('aria-label', `Remove ${doc.name}`);
    remove.onclick = () => task(async () => {
      await api(`/api/documents/${encodeURIComponent(doc.id)}`, {method: 'DELETE'});
      $('answer').replaceChildren(element('p', 'Library updated. Ask again to search your current notes.', 'answer-text'));
      await refresh();
    });
    row.append(info, remove); $('documents').append(row);
  }
}
async function task(action) {
  if (busy) return;
  busy = true; $('message').textContent = 'Working…';
  document.querySelectorAll('button,input').forEach(n => n.disabled = true);
  try { await action(); $('message').textContent = ''; }
  catch (error) { $('message').textContent = error.message; }
  finally {
    busy = false; document.querySelectorAll('button,input').forEach(n => n.disabled = false);
  }
}
$('demo').onclick = () => task(async () => { await api('/api/demo', {method: 'POST'}); await refresh(); });
$('file').onchange = () => task(async () => {
  if (!$('file').files.length) return;
  const form = new FormData(); form.append('file', $('file').files[0]);
  await api('/api/documents', {method: 'POST', body: form});
  $('file').value = ''; await refresh();
});
$('ask').onsubmit = (event) => {
  event.preventDefault();
  task(async () => {
    const data = await api('/api/ask', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: $('question').value, generate: $('generate').checked})});
    $('answer').replaceChildren(element('div', data.generated ? 'LOCAL AI EXPLANATION · VERIFY AGAINST SOURCES' : 'RETRIEVED PASSAGES · NO GENERATED ANSWER', 'answer-label'), element('div', data.answer, 'answer-text'));
    if (data.warning) $('answer').append(element('p', data.warning, 'warning'));
    for (const source of data.sources) {
      const card = element('details', undefined, 'source');
      card.append(element('summary', `[${source.citation}] ${source.name} · page ${source.page}`), element('p', source.text));
      $('answer').append(card);
    }
    $('retrieval-mode').textContent = data.mode === 'hybrid' ? '◉ Hybrid semantic retrieval' : '▧ Lexical retrieval';
  });
};
document.querySelectorAll('[data-question]').forEach(button => button.onclick = () => {
  $('question').value = button.dataset.question; $('question').focus();
});
Promise.all([refresh(), api('/api/health').then(status => {
  $('retrieval-mode').textContent = status.semantic ? '◉ Hybrid semantic retrieval' : '▧ Lexical retrieval';
})]).catch(error => $('message').textContent = error.message);
