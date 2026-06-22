(function () {
  const storageKey = "tb_theme";
  const root = document.documentElement;

  function getInitialTheme() {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved === "light" || saved === "dark") {
        return saved;
      }
    } catch (error) {
      // Ignore storage access errors and fall back to system preference.
    }

    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    return prefersDark ? "dark" : "light";
  }

  function setTheme(theme, persist) {
    root.setAttribute("data-theme", theme);

    if (persist) {
      try {
        localStorage.setItem(storageKey, theme);
      } catch (error) {
        // Ignore storage access errors to avoid breaking UI interaction.
      }
    }
  }

  function syncToggleButtons(isDark) {
    const title = isDark ? "Switch to light mode" : "Switch to dark mode";

    document.querySelectorAll("[data-theme-toggle]").forEach(function (button) {
      button.setAttribute("title", title);
      button.setAttribute("aria-label", title);
      button.setAttribute("aria-pressed", String(isDark));
      button.setAttribute("data-mode", isDark ? "dark" : "light");
    });
  }

  setTheme(getInitialTheme(), false);

  document.addEventListener("DOMContentLoaded", function () {
    const toggles = document.querySelectorAll("[data-theme-toggle]");

    if (!toggles.length) {
      return;
    }

    syncToggleButtons(root.getAttribute("data-theme") === "dark");

    toggles.forEach(function (button) {
      button.addEventListener("click", function () {
        const isDark = root.getAttribute("data-theme") === "dark";
        const nextTheme = isDark ? "light" : "dark";

        setTheme(nextTheme, true);
        syncToggleButtons(nextTheme === "dark");
      });
    });
  });
})();
