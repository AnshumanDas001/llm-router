/* Shared nav, injected into any page with <div id="app-nav-top-mount"> and
   <div id="app-nav-bottom-mount"> inside a <nav class="app-nav">. Split into
   two mount points (rather than one) so a page can put its own content
   (e.g. the chat app's conversation list) between the nav links and the
   user/logout footer, while still sharing the same links/auth logic. */

const NAV_ICONS = {
  chat: '<svg viewBox="0 0 16 16" fill="none"><path d="M2 3.5h12v7H6.5L3 13.5V10.5H2v-7Z" stroke="currentColor" stroke-width="1.3" stroke-linejoin="round"/></svg>',
  sessions: '<svg viewBox="0 0 16 16" fill="none"><path d="M2.5 13.5v-4M7 13.5V6M11.5 13.5V3" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/></svg>',
  models: '<svg viewBox="0 0 16 16" fill="none"><path d="M8 2.2 13.5 5 8 7.8 2.5 5 8 2.2Z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/><path d="M2.5 8.2 8 11l5.5-2.8M2.5 11.3 8 14.1l5.5-2.8" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>',
  settings: '<svg viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="2.1" stroke="currentColor" stroke-width="1.3"/><path d="M8 2.3v1.6M8 12.1v1.6M2.3 8h1.6M12.1 8h1.6M4 4l1.1 1.1M10.9 10.9 12 12M12 4l-1.1 1.1M5.1 10.9 4 12" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>',
  docs: '<svg viewBox="0 0 16 16" fill="none"><path d="M2.5 3.2c1.6-.6 3.4-.5 5 .3v9.3c-1.6-.8-3.4-.9-5-.3V3.2ZM13.5 3.2c-1.6-.6-3.4-.5-5 .3v9.3c1.6-.8 3.4-.9 5-.3V3.2Z" stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/></svg>',
};

async function initNav(activePage) {
  const topMount = document.getElementById("app-nav-top-mount");
  const bottomMount = document.getElementById("app-nav-bottom-mount");
  if (!topMount || !bottomMount) return;

  const links = [
    { key: "chat", href: "/app", label: "Chat" },
    { key: "sessions", href: "/sessions", label: "Sessions" },
    { key: "models", href: "/models", label: "Models" },
    { key: "settings", href: "/settings", label: "Settings" },
    { key: "docs", href: "/guide", label: "Docs" },
  ];

  const linksHtml = links.map(l => `
    <a class="app-nav-link${l.key === activePage ? " active" : ""}" href="${l.href}">
      <span class="nav-icon">${NAV_ICONS[l.key]}</span>
      <span class="nav-label">${l.label}</span>
      <span class="nav-badge" data-nav-badge="${l.key}" hidden></span>
    </a>
  `).join("");

  // A page may provide its own links mount (the chat app puts the "new chat"
  // button between the brand and the links); otherwise links follow the brand.
  const linksMount = document.getElementById("app-nav-links-mount");
  topMount.innerHTML = `
    <div class="brand"><span class="brand-name">Thrift<span class="dot">LLM</span></span><span class="brand-version">v1</span></div>
    ${linksMount ? "" : linksHtml}
  `;
  if (linksMount) linksMount.innerHTML = linksHtml;

  bottomMount.innerHTML = `
    <div class="app-nav-user">
      <span class="nav-avatar" id="nav-avatar"></span>
      <span class="nav-user-text"><span id="nav-username"></span></span>
      <button id="nav-logout-btn" title="Log out">
        <svg viewBox="0 0 16 16" fill="none"><path d="M6.5 13.5H3.5v-11h3M10 10.5 13 8l-3-2.5M13 8H6" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
    </div>
  `;

  try {
    const resp = await fetch("/auth/me", { credentials: "same-origin" });
    if (!resp.ok) throw new Error("not authenticated");
    const data = await resp.json();
    document.getElementById("nav-username").textContent = data.username;
    document.getElementById("nav-username").title = data.username;
    document.getElementById("nav-avatar").textContent = (data.username[0] || "?").toUpperCase();
  } catch {
    window.location.href = "/app";
    return;
  }

  document.getElementById("nav-logout-btn").onclick = async () => {
    await fetch("/auth/logout", { method: "POST", credentials: "same-origin" });
    window.location.href = "/app";
  };

  // Real (not simulated) total saved across every session, shown as a
  // small badge on the Sessions link -- same numbers sessions.html itself
  // shows, just summed.
  try {
    const resp = await fetch("/api/sessions", { credentials: "same-origin" });
    if (resp.ok) {
      const sessions = await resp.json();
      const totalSaved = sessions.reduce((sum, s) => sum + (s.cost_saved || 0), 0);
      const badge = document.querySelector('[data-nav-badge="sessions"]');
      if (badge && totalSaved > 0) {
        badge.textContent = "+$" + totalSaved.toFixed(2);
        badge.hidden = false;
      }
    }
  } catch {
    /* non-critical decoration -- leave the badge hidden on failure */
  }
}
