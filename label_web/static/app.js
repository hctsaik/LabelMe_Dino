const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const meta = document.getElementById("meta");
const cls = document.getElementById("cls");

let task = { images: [], classes: ["object"] };
let index = 0;
let img = new Image();
let shapes = [];
let drag = null;
let natural = { w: 0, h: 0 };
let dirty = false;
let view = { ox: 0, oy: 0, k: 1 };

async function api(path, opts) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = data.message || data.error || r.statusText;
    throw new Error(msg);
  }
  return data;
}

function fitCanvas() {
  const wrap = document.getElementById("stage-wrap");
  const dw = Math.max(1, wrap.clientWidth);
  const dh = Math.max(1, wrap.clientHeight);
  cv.width = dw;
  cv.height = dh;
  cv.style.width = dw + "px";
  cv.style.height = dh + "px";
  if (!natural.w) {
    view = { ox: 0, oy: 0, k: 1 };
    return;
  }
  const k = Math.min(dw / natural.w, dh / natural.h);
  const w = natural.w * k;
  const h = natural.h * k;
  view = { ox: (dw - w) / 2, oy: (dh - h) / 2, k };
}

function toImage(clientX, clientY) {
  const r = cv.getBoundingClientRect();
  const x = (clientX - r.left - view.ox) / view.k;
  const y = (clientY - r.top - view.oy) / view.k;
  return {
    x: Math.max(0, Math.min(natural.w, x)),
    y: Math.max(0, Math.min(natural.h, y)),
  };
}

function highlightQueue() {
  document.querySelectorAll("#queue button").forEach((b, i) => {
    b.classList.toggle("active", i === index);
  });
}

function redraw() {
  ctx.fillStyle = "#0b0e14";
  ctx.fillRect(0, 0, cv.width, cv.height);
  if (img.src && natural.w) {
    ctx.drawImage(img, view.ox, view.oy, natural.w * view.k, natural.h * view.k);
  }
  const s = view.k;
  ctx.lineWidth = 2;
  ctx.font = "12px sans-serif";
  for (const sh of shapes) {
    const pts = sh.points || [];
    if (pts.length < 2) continue;
    const x = view.ox + pts[0][0] * s;
    const y = view.oy + pts[0][1] * s;
    const w = (pts[1][0] - pts[0][0]) * s;
    const h = (pts[1][1] - pts[0][1]) * s;
    ctx.strokeStyle = "#7ec2ff";
    ctx.fillStyle = "rgba(80,160,255,0.12)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#7ec2ff";
    ctx.fillText(sh.label || "", x, Math.max(12, y - 4));
  }
  if (drag) {
    ctx.strokeStyle = "#ffcc66";
    ctx.strokeRect(
      Math.min(drag.x0, drag.x1),
      Math.min(drag.y0, drag.y1),
      Math.abs(drag.x1 - drag.x0),
      Math.abs(drag.y1 - drag.y0)
    );
  }
}

function setEmpty(on) {
  document.getElementById("empty").hidden = !on;
  document.getElementById("stage-wrap").hidden = on;
}

async function load(i) {
  if (!task.images.length) {
    setEmpty(true);
    meta.textContent = "尚未開啟資料";
    return;
  }
  setEmpty(false);
  if (dirty) {
    const ok = window.confirm("有未儲存的框。確定丟棄並切圖？");
    if (!ok) return;
  }
  index = (i + task.images.length) % task.images.length;
  const info = task.images[index];
  meta.textContent = `${index + 1} / ${task.images.length}  ${info.name}`;
  meta.className = "";
  const ann = await api("/api/annotation/" + index);
  shapes = ann.shapes || [];
  dirty = false;
  img = new Image();
  img.onload = () => {
    natural = { w: img.naturalWidth, h: img.naturalHeight };
    requestAnimationFrame(() => {
      fitCanvas();
      redraw();
      highlightQueue();
    });
  };
  img.src = "/api/image/" + index + "?t=" + Date.now();
}

async function save() {
  try {
    await api("/api/annotation/" + index, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        shapes,
        imageWidth: natural.w,
        imageHeight: natural.h,
      }),
    });
    dirty = false;
    meta.textContent = `${index + 1} / ${task.images.length}  ${task.images[index].name} · 已存`;
    meta.className = "";
  } catch (err) {
    meta.textContent = String(err.message || err);
    meta.className = "error";
  }
}

cv.addEventListener("mousedown", (e) => {
  const r = cv.getBoundingClientRect();
  drag = { x0: e.clientX - r.left, y0: e.clientY - r.top, x1: 0, y1: 0 };
  drag.x1 = drag.x0;
  drag.y1 = drag.y0;
});
cv.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const r = cv.getBoundingClientRect();
  drag.x1 = e.clientX - r.left;
  drag.y1 = e.clientY - r.top;
  redraw();
});
window.addEventListener("mouseup", () => {
  if (!drag) return;
  const a = toImage(drag.x0 + cv.getBoundingClientRect().left, drag.y0 + cv.getBoundingClientRect().top);
  const b = toImage(drag.x1 + cv.getBoundingClientRect().left, drag.y1 + cv.getBoundingClientRect().top);
  const x0 = Math.min(a.x, b.x);
  const y0 = Math.min(a.y, b.y);
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  if (Math.abs(x1 - x0) > 4 && Math.abs(y1 - y0) > 4) {
    shapes.push({
      label: cls.value || "object",
      points: [
        [x0, y0],
        [x1, y1],
      ],
      shape_type: "rectangle",
    });
    dirty = true;
  }
  drag = null;
  redraw();
});

document.getElementById("prev").onclick = () => load(index - 1);
document.getElementById("next").onclick = () => load(index + 1);
document.getElementById("del").onclick = () => {
  shapes.pop();
  dirty = true;
  redraw();
};
document.getElementById("save").onclick = save;

function applyTask(next) {
  task = next;
  setEmpty(!(task.images && task.images.length));
  cls.innerHTML = "";
  for (const c of task.classes || ["object"]) {
    const o = document.createElement("option");
    o.value = c;
    o.textContent = c;
    cls.appendChild(o);
  }
  const queue = document.getElementById("queue");
  queue.innerHTML = "";
  (task.images || []).forEach((info, i) => {
    const b = document.createElement("button");
    b.type = "button";
    const im = document.createElement("img");
    im.src = "/api/image/" + i + "?t=" + Date.now();
    im.alt = info.name;
    b.appendChild(im);
    b.onclick = () => load(i);
    if (i === index) b.className = "active";
    queue.appendChild(b);
  });
}

async function pollJob(jobId) {
  const jobEl = document.getElementById("job");
  for (;;) {
    const rec = await api("/api/jobs/" + jobId);
    const p = rec.progress || {};
    jobEl.textContent = `${rec.kind} ${rec.status} ${p.stage || ""} ${p.done || 0}/${p.total || 0}`;
    if (rec.status === "succeeded") {
      if (rec.task) applyTask(rec.task);
      await load(index);
      return rec;
    }
    if (rec.status === "failed") throw new Error(rec.error || "job failed");
    await new Promise((r) => setTimeout(r, 400));
  }
}

document.getElementById("open-folder").onclick = async () => {
  try {
    const r = await api("/api/host/pick-folder", { method: "POST", body: "{}" });
    applyTask(r.task);
    await load(0);
  } catch (err) {
    meta.textContent = String(err.message || err);
  }
};
document.getElementById("open-video").onclick = async () => {
  try {
    const r = await api("/api/host/pick-video", { method: "POST", body: "{}" });
    document.getElementById("video-path").value = r.video_path;
    document.getElementById("prepare").click();
  } catch (err) {
    meta.textContent = String(err.message || err);
  }
};
document.getElementById("prepare").onclick = async () => {
  try {
    const started = await api("/api/jobs/prepare_video", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_path: document.getElementById("video-path").value,
        start_sec: Number(document.getElementById("t0").value),
        end_sec: Number(document.getElementById("t1").value),
      }),
    });
    await pollJob(started.job_id);
  } catch (err) {
    document.getElementById("job").textContent = String(err.message || err);
  }
};

document.getElementById("propagate").onclick = async () => {
  try {
    await save();
    const name = (task.images[index] || {}).stem || "";
    const m = /frame_(\d+)/.exec(name);
    const anchor = m ? Number(m[1]) : index;
    const started = await api("/api/jobs/propagate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        anchor_frame: anchor,
        shapes,
        classes: task.classes,
        radius_seconds: 2,
      }),
    });
    await pollJob(started.job_id);
  } catch (err) {
    document.getElementById("job").textContent = String(err.message || err);
  }
};

document.getElementById("sam").onclick = async () => {
  const last = shapes.filter((s) => s.shape_type === "rectangle").slice(-1)[0];
  if (!last || !last.points || last.points.length < 2) {
    meta.textContent = "請先畫一個框再分割";
    return;
  }
  const xs = last.points.map((p) => p[0]);
  const ys = last.points.map((p) => p[1]);
  try {
    const r = await api("/api/sam/propose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        index,
        roi: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)],
      }),
    });
    const box = document.getElementById("cands");
    box.hidden = false;
    box.innerHTML = "";
    for (const c of r.candidates || []) {
      const b = document.createElement("button");
      b.type = "button";
      const im = document.createElement("img");
      im.src = "/api/sam/" + c.candidate_id + "/png";
      b.appendChild(im);
      b.onclick = async () => {
        await api("/api/sam/accept", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ candidate_id: c.candidate_id }),
        });
        meta.textContent = "已 Accept 遮罩";
      };
      box.appendChild(b);
    }
  } catch (err) {
    meta.textContent = String(err.message || err);
  }
};

window.addEventListener("resize", () => {
  fitCanvas();
  redraw();
});
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === "n" || e.key === "N" || e.key === "ArrowRight") load(index + 1);
  if (e.key === "p" || e.key === "P" || e.key === "ArrowLeft") load(index - 1);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    save();
  }
});

(async () => {
  try {
    task = await api("/api/task");
  } catch (err) {
    meta.textContent = String(err.message || err);
    meta.className = "error";
    return;
  }
  applyTask(task);
  await load(0);
})();
