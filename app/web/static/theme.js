/* Light / dark theme. Loaded in <head> so the saved choice applies before
   the first paint. Light is the default; the toggle saves a choice of
   dark (or back to light) in this browser.

   Any element with [data-theme-toggle] becomes a toggle button. */
(function () {
  var KEY = "tl_theme";
  var root = document.documentElement;
  var SUN = '<svg viewBox="0 0 16 16" width="15" height="15" fill="none" aria-hidden="true"><circle cx="8" cy="8" r="3" stroke="currentColor" stroke-width="1.4"/><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M12.6 3.4l-1.1 1.1M4.5 11.5l-1.1 1.1" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
  var MOON = '<svg viewBox="0 0 16 16" width="15" height="15" fill="none" aria-hidden="true"><path d="M13.5 9.6A5.6 5.6 0 0 1 6.4 2.5a5.6 5.6 0 1 0 7.1 7.1Z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/></svg>';

  function saved() { try { return localStorage.getItem(KEY); } catch (e) { return null; } }
  root.setAttribute("data-theme", saved() === "dark" ? "dark" : "light");

  function current() { return root.getAttribute("data-theme") === "dark" ? "dark" : "light"; }

  function paint(btn) {
    var dark = current() === "dark";
    // the icon shows where the toggle goes, not where you are
    btn.innerHTML = dark ? SUN : MOON;
    btn.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    btn.title = btn.getAttribute("aria-label");
  }

  function toggle() {
    var next = current() === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem(KEY, next); } catch (e) { /* private mode: applies for this page only */ }
    document.querySelectorAll("[data-theme-toggle]").forEach(paint);
  }

  window.mountThemeToggles = function () {
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      if (btn.dataset.themeBound) return paint(btn);
      btn.dataset.themeBound = "1";
      btn.type = "button";
      btn.addEventListener("click", toggle);
      paint(btn);
    });
  };
  document.addEventListener("DOMContentLoaded", window.mountThemeToggles);
})();
