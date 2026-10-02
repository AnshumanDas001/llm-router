/* After a cold start the router's language model takes up to a minute to
   load (torch and an embedding model on a small CPU). The pages are served
   straight away; this shows a notice until /api/status says the router is
   ready, then announces it.

   Pages can wait on it:  window.routerReady  (boolean)
                          window.onRouterReady(callback)
   and the "router-ready" event fires on window. */
(function () {
  let ready = false;
  const waiters = [];
  window.routerReady = false;
  window.onRouterReady = function (cb) { if (ready) cb(); else waiters.push(cb); };

  function markReady() {
    if (ready) return;
    ready = true;
    window.routerReady = true;
    waiters.splice(0).forEach(cb => { try { cb(); } catch (e) { console.error(e); } });
    window.dispatchEvent(new Event("router-ready"));
  }

  let toast = null, timer = null, shownAt = 0;
  function showLoading() {
    if (toast) return;
    shownAt = Date.now();
    toast = document.createElement("div");
    toast.className = "warmup-toast";
    toast.setAttribute("role", "status");
    toast.innerHTML = `<span class="spinner"></span>
      <div><b>Starting the router</b>
      <span>Loading the language model that sorts prompts by difficulty. After the app has been idle this takes about a minute. You can look around meanwhile; sending unlocks when it's ready.</span></div>
      <span class="warmup-time mono">0s</span>`;
    document.body.appendChild(toast);
    timer = setInterval(() => {
      const el = toast && toast.querySelector(".warmup-time");
      if (el) el.textContent = `${Math.round((Date.now() - shownAt) / 1000)}s`;
    }, 1000);
  }
  function showReady() {
    if (!toast) return;          // it was ready before anyone had to wait
    clearInterval(timer);
    toast.classList.add("done");
    toast.innerHTML = `<span class="ok">&#10003;</span><div><b>Router ready</b>
      <span>Loaded in ${Math.round((Date.now() - shownAt) / 1000)}s. Go ahead.</span></div>`;
    setTimeout(() => { toast && toast.remove(); toast = null; }, 4000);
  }
  function showFailed(msg) {
    if (!toast) showLoading();
    clearInterval(timer);
    toast.classList.add("failed");
    toast.innerHTML = `<span class="bad">!</span><div><b>The router didn't start</b>
      <span>${msg ? msg.replace(/</g, "&lt;") : "Reload the page in a moment."}</span></div>`;
  }

  async function poll(first) {
    try {
      // A long poll: the server holds the request until the model is loaded
      // or 20s pass. On a scale-to-zero host that open request is also what
      // keeps the CPU allocated while the model loads.
      const resp = await fetch(`/api/status?wait=${first ? 0 : 20}`, { cache: "no-store" });
      const s = await resp.json();
      if (s.ready) { showReady(); markReady(); return; }
      if (s.stage === "failed") { showFailed(s.error); return; }
      showLoading();
    } catch {
      showLoading();
      await new Promise(r => setTimeout(r, 3000));
    }
    poll(false);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => poll(true));
  else poll(true);
})();
