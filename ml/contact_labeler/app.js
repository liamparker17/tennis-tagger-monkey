const FPS = 30;
let state = { matches: [], match: null, clip: null, events: [] };

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

function loadClip(match, clip) {
  state.match = match; state.clip = clip; state.events = [];
  document.getElementById("v").src = `/clip/${match}/${clip}.mp4`;
  renderSide(); renderEvents();
}

function frame() { return Math.round(document.getElementById("v").currentTime * FPS); }
function step(d) { document.getElementById("v").currentTime = Math.max(0, frame()/FPS + d/FPS); }

function mark(hitter) {
  state.events.push({ frame: frame(), hitter });
  renderEvents();
}

function undo() { state.events.pop(); renderEvents(); }

async function save() {
  if (!state.match || !state.clip) return;
  await fetch(`/api/labels/${state.match}`, {
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

function renderEvents() {
  document.getElementById("bar").textContent =
    `${state.match || "-"} / ${state.clip || "-"}  frame=${frame()}  events=${state.events.length}`;
  document.getElementById("events").innerHTML =
    state.events.map(e => `f=${e.frame} hitter=${e.hitter}`).join("<br>");
}

document.addEventListener("keydown", (e) => {
  const v = document.getElementById("v");
  if (e.key === ",") { v.pause(); step(-1); }
  else if (e.key === ".") { v.pause(); step(1); }
  else if (e.key === " ") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === "n") mark(0);
  else if (e.key === "f") mark(1);
  else if (e.key === "u") undo();
  else if (e.key === "s") save();
  else if (e.key === "]") nextClip();
});
document.getElementById("v").addEventListener("timeupdate", renderEvents);

loadMatches();
