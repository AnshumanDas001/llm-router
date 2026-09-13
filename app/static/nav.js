/* Shared nav, injected into any page with <div id="app-nav-top-mount"> and
   <div id="app-nav-bottom-mount"> inside a <nav class="app-nav">. Split into
   two mount points (rather than one) so a page can put its own content
   (e.g. the chat app's conversation list) between the nav links and the
   user/logout footer, while still sharing the same links/auth logic. */
async function initNav(activePage) {
  const topMount = document.getElementById("app-nav-top-mount");
  const bottomMount = document.getElementById("app-nav-bottom-mount");
  if (!topMount || !bottomMount) return;

  const links = [
    { key: "chat", href: "/app", label: "Chat" },
    { key: "sessions", href: "/sessions", label: "Sessions" },
    { key: "settings", href: "/settings", label: "Settings" },
    { key: "docs", href: "/docs", label: "Docs" },
  ];

  topMount.innerHTML = `
    <div class="brand">LLM<span class="dot">Router</span></div>
    ${links.map(l => `<a class="app-nav-link${l.key === activePage ? " active" : ""}" href="${l.href}">${l.label}</a>`).join("")}
  `;
  bottomMount.innerHTML = `
    <div class="app-nav-user">
      <span id="nav-username"></span>
      <button id="nav-logout-btn">log out</button>
    </div>
  `;

  try {
    const resp = await fetch("/auth/me", { credentials: "same-origin" });
    if (!resp.ok) throw new Error("not authenticated");
    const data = await resp.json();
    document.getElementById("nav-username").textContent = data.username;
  } catch {
    window.location.href = "/app";
    return;
  }

  document.getElementById("nav-logout-btn").onclick = async () => {
    await fetch("/auth/logout", { method: "POST", credentials: "same-origin" });
    window.location.href = "/app";
  };
}
