const FPS = 30;
let state = {
  matches: [], match: null, clip: null, events: [],
  // Populated from /api/clip_info on loadClip.
  info: { player_a: "", player_b: "", near_name: "", far_name: "",
          near_is_p1: true, stroke_types: [], stroke_count_lo: 0, stroke_count_hi: 0 },
};

async function loadMatches() {
  const r = await fetch("/api/matches");
  state.matches = (await r.json()).matches;
  renderSide();
}

function renderSide() {
  const el = document.getElementById("side");
  el.innerHTML = "";
  for (const m of state.matches) {
    const h = document.createElement("div"); h.className = "m"; h.textContent = m.name; el.appendChild(h);
    for (const c of m.clips) {
      const d = document.createElement("div"); d.className = "c";
      if (m.labeled.includes(c)) d.classList.add("done");
      if (state.match === m.name && state.clip === c) d.classList.add("active");
      d.textContent = c;
      d.onclick = () => loadClip(m.name, c);
      el.appendChild(d);
    }
  }
}

async function loadClip(match, clip) {
  state.match = match; state.clip = clip; state.events = [];
  document.getElementById("v").src = `/clip/${match}/${clip}.mp4`;
  // Fetch saved events + player-name context for this clip.
  try {
    const r = await fetch(`/api/clip_info/${encodeURIComponent(match)}/${encodeURIComponent(clip)}`);
    state.info = await r.json();
    state.events = Array.isArray(state.info.events) ? state.info.events.slice() : [];
  } catch (_) { /* keep defaults */ }
  renderSide(); renderEvents();
}

function frame() { return Math.round(document.getElementById("v").currentTime * FPS); }
function step(d) { document.getElementById("v").currentTime = Math.max(0, frame()/FPS + d/FPS); }

// `n` = near-side player hit it. Map to hitter index 0 (=player_a) if the
// near player is currently player_a at this clip, else 1.
function mark(side) {
  const nearIsP1 = !!state.info.near_is_p1;
  // side === "near" → hitter = nearIsP1 ? 0 : 1; side === "far" → opposite
  const hitter = (side === "near") ? (nearIsP1 ? 0 : 1) : (nearIsP1 ? 1 : 0);
  state.events.push({ frame: frame(), hitter });
  state.events.sort((a, b) => a.frame - b.frame);
  renderEvents();
}

function undo() { state.events.pop(); renderEvents(); }

async function save() {
  if (!state.match || !state.clip) return;
  await fetch(`/api/labels/${encodeURIComponent(state.match)}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clip: state.clip, events: state.events }),
  });
  await loadMatches();
}

function nextClip() {
  const m = state.matches.find(x => x.name === state.match);
  if (!m) return;
  const i = m.clips.indexOf(state.clip);
  if (i >= 0 && i + 1 < m.clips.length) loadClip(m.name, m.clips[i+1]);
}

function hitterName(hitter) {
  if (hitter === 0) return state.info.player_a || "P1";
  if (hitter === 1) return state.info.player_b || "P2";
  return `h=${hitter}`;
}

function renderEvents() {
  const info = state.info;
  const nearLbl = info.near_name || (info.near_is_p1 ? (info.player_a || "P1") : (info.player_b || "P2"));
  const farLbl = info.far_name || (info.near_is_p1 ? (info.player_b || "P2") : (info.player_a || "P1"));
  const strokeHint = info.stroke_types && info.stroke_types.length
    ? ` · Dartfish: ${info.stroke_types.length} strokes tagged (bucket ${info.stroke_count_lo}–${info.stroke_count_hi})`
    : "";
  document.getElementById("bar").textContent =
    `${state.match || "-"} / ${state.clip || "-"}  frame=${frame()}  events=${state.events.length}${strokeHint}`;
  document.getElementById("players").innerHTML =
    `Press <b>n</b> for NEAR hit: <b>${escapeHtml(nearLbl)}</b> &nbsp;|&nbsp; ` +
    `Press <b>f</b> for FAR hit: <b>${escapeHtml(farLbl)}</b>`;
  document.getElementById("events").innerHTML =
    state.events.map(e => `f=${e.frame} hitter=${e.hitter} (${escapeHtml(hitterName(e.hitter))})`).join("<br>");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c]));
}

document.addEventListener("keydown", (e) => {
  const v = document.getElementById("v");
  if (e.key === ",") { v.pause(); step(-1); }
  else if (e.key === ".") { v.pause(); step(1); }
  else if (e.key === " ") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === "n") mark("near");
  else if (e.key === "f") mark("far");
  else if (e.key === "u") undo();
  else if (e.key === "s") save();
  else if (e.key === "]") nextClip();
});
document.getElementById("v").addEventListener("timeupdate", renderEvents);

loadMatches();
