// ============================================================
// main.js — 应用入口 + Tab 切换 + URL hash 状态持久化
// ============================================================

const APP = (function () {
  let params = null;

  function initParams() {
    params = MODEL.initParams(7);
  }

  function resetParams() {
    initParams();
    if (window.ComponentsView && ComponentsView.init) ComponentsView.init(params);
    if (window.InferenceView && InferenceView.refresh) InferenceView.refresh();
  }

  function getParams() { return params; }

  function switchTab(name, skipHash) {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("active", t.dataset.tab === name);
    });
    document.querySelectorAll(".view").forEach((v) => {
      v.classList.toggle("active", v.id === `view-${name}`);
    });
    if (!skipHash) updateHash(name, null);
  }

  function updateHash(tab, step) {
    const h = step != null ? `${tab}/step=${step}` : tab;
    history.replaceState(null, "", `#${h}`);
  }

  function readHash() {
    const raw = location.hash.slice(1);
    if (!raw) return null;
    const parts = raw.split("/");
    const tab = parts[0];
    const stepPart = parts.find((p) => p.startsWith("step="));
    return { tab, step: stepPart ? parseInt(stepPart.split("=")[1]) : null };
  }

  function bindTabs() {
    document.querySelectorAll(".tab").forEach((t) => {
      t.addEventListener("click", () => switchTab(t.dataset.tab));
    });
  }

  function init() {
    initParams();
    bindTabs();

    ArchitectureView.init((target) => {
      switchTab("components");
      ComponentsView.jumpFromArch(target);
    });
    ComponentsView.init(params);
    TrainingView.init(params);
    InferenceView.init(params);

    // 恢复上次的 tab/step
    const saved = readHash();
    const validTabs = ["architecture", "components", "training", "inference"];
    if (saved && validTabs.includes(saved.tab)) {
      switchTab(saved.tab, true);
      if (saved.tab === "components" && saved.step != null) {
        ComponentsView.jumpFromArch("__step__" + saved.step);
      }
    }

    // 当 components 步骤改变时同步 hash
    document.getElementById("step-prev").addEventListener("click", syncStepHash, { capture: true });
    document.getElementById("step-next").addEventListener("click", syncStepHash, { capture: true });
    document.getElementById("step-nav").addEventListener("click", syncStepHash, { capture: true });
  }

  function syncStepHash() {
    // 延迟一帧等 jumpTo 完成
    requestAnimationFrame(() => {
      const indEl = document.getElementById("step-indicator");
      if (!indEl) return;
      const cur = parseInt(indEl.textContent) - 1;
      updateHash("components", isNaN(cur) ? 0 : cur);
    });
  }

  return { init, getParams, resetParams };
})();

window.APP = APP;
document.addEventListener("DOMContentLoaded", APP.init);
