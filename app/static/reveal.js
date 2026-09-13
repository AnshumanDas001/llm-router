/* Scroll-reveal: adds .in-view to .reveal elements as they enter the
   viewport. Elements are revealed immediately (and the observer skipped)
   when the viewer prefers reduced motion or IntersectionObserver is
   unavailable, so content is never left invisible. */
(function () {
  function revealAll() {
    document.querySelectorAll(".reveal").forEach(el => el.classList.add("in-view"));
  }
  const reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced || !("IntersectionObserver" in window)) { revealAll(); return; }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in-view");
        observer.unobserve(entry.target);
      }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });

  function observe() { document.querySelectorAll(".reveal:not(.in-view)").forEach(el => observer.observe(el)); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", observe);
  else observe();
  window.revealObserve = observe;   // for content added after load
})();
