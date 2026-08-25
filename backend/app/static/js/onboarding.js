/* ==========================================================================
   Czargroup onboarding

   Renders the platform/product pickers from /api/onboarding/catalog, keeps the
   licence panel in step with the form, and posts the signup.

   The catalogue is fetched rather than templated so adding a product to
   catalog.PRODUCTS on the server puts it on the page with no front-end change.
   That matters because the product list is the thing most likely to grow.

   No framework and no build step: this page is four fieldsets and a preview
   panel, and a dependency chain would cost more than it saves.
   ========================================================================== */

(function () {
  "use strict";

  var EM_DASH = "—";

  var state = {
    catalog: null,
    platform: null,   // platform object from the catalogue
    product: null,    // product object
    plan: null,       // plan object
    issued: false,
  };

  var el = {};

  function $(id) { return document.getElementById(id); }

  function cacheNodes() {
    el.platformChoices = $("platform-choices");
    el.productChoices  = $("product-choices");
    el.productStep     = $("step-product");
    el.planChoices     = $("plan-choices");

    el.form      = $("signup-form");
    el.error     = $("form-error");
    el.submit    = $("submit-btn");
    el.storeUrl  = $("store-url");

    el.licence       = $("licence");
    el.licenceBar    = $("licence-bar");
    el.licenceStamp  = $("licence-stamp");
    el.licenceExpiry = $("licence-expiry");

    el.issued        = $("issued");
    el.issuedTitle   = $("issued-title");
    el.issuedLede    = $("issued-lede");
    el.keyValue      = $("key-value");
    el.copyBtn       = $("copy-btn");
    el.installHead   = $("install-heading");
    el.installSteps  = $("install-steps");
    el.anotherBtn    = $("another-btn");
  }

  /* ── Accent ─────────────────────────────────────────────────────────────
     The page's accent colour is the selected platform's own colour. Setting it
     on the root element means every rule that references var(--accent) — the
     step numbers, the selection bars, the licence panel edge, the submit
     button — recolours together from one assignment. */
  function setAccent(color) {
    document.documentElement.style.setProperty("--accent", color || "");
  }

  /* ── Licence panel ──────────────────────────────────────────────────────
     Values are written through a single helper so the "not yet filled" styling
     is decided in one place rather than at each call site. */
  function setPanelField(name, value) {
    var node = document.querySelector('[data-licence-field="' + name + '"]');
    if (!node) return;
    var filled = value !== null && value !== undefined && value !== "";
    node.textContent = filled ? value : EM_DASH;
    node.setAttribute("data-empty", filled ? "false" : "true");
  }

  /* Best-effort mirror of extract_domain() on the server. Only used for the
     live preview — the server's answer is what actually goes on the key, and
     this never overrides it. Kept deliberately forgiving: someone mid-typing
     shouldn't see an error, just an unfilled field. */
  function previewDomain(raw) {
    var value = (raw || "").trim();
    if (!value) return "";
    if (value.indexOf("://") === -1) value = "https://" + value;
    try {
      var host = new URL(value).hostname.toLowerCase();
      if (host.indexOf("www.") === 0) host = host.slice(4);
      return host;
    } catch (err) {
      return "";
    }
  }

  function refreshPanel() {
    setPanelField("platform", state.platform ? state.platform.name : "");
    setPanelField("product",  state.product  ? state.product.name  : "");
    setPanelField("code",     state.product  ? state.product.code  : "");
    setPanelField("domain",   previewDomain(el.storeUrl.value));
    setPanelField("plan",     state.plan ? state.plan.name : "");

    if (!state.issued) {
      el.licenceExpiry.textContent = "365 days from issue";
      el.licenceExpiry.setAttribute("data-empty", "true");
    }
  }

  /* ── Rendering ──────────────────────────────────────────────────────────── */

  function platformCard(platform) {
    var label = document.createElement("label");
    label.className = "choice";

    var input = document.createElement("input");
    input.type = "radio";
    input.name = "platform";
    input.value = platform.code;

    var body = document.createElement("span");
    body.className = "choice__body";
    body.innerHTML =
      '<span class="choice__name"></span>' +
      '<span class="choice__tagline"></span>';
    body.querySelector(".choice__name").textContent = platform.name;
    body.querySelector(".choice__tagline").textContent = platform.blurb;

    input.addEventListener("change", function () {
      selectPlatform(platform);
    });

    label.appendChild(input);
    label.appendChild(body);
    return label;
  }

  function productCard(product) {
    var label = document.createElement("label");
    label.className = "choice";

    var input = document.createElement("input");
    input.type = "radio";
    input.name = "product_code";
    input.value = product.code;
    input.required = true;

    var body = document.createElement("span");
    body.className = "choice__body";
    body.innerHTML =
      '<span class="choice__name"></span>' +
      '<span class="choice__tagline"></span>' +
      '<span class="choice__artifact"></span>';
    body.querySelector(".choice__name").textContent = product.name;
    body.querySelector(".choice__tagline").textContent = product.tagline;
    body.querySelector(".choice__artifact").textContent = product.artifact;

    input.addEventListener("change", function () {
      state.product = product;
      refreshPanel();
      updateSubmitState();
    });

    label.appendChild(input);
    label.appendChild(body);
    return label;
  }

  function planCard(plan) {
    var label = document.createElement("label");
    label.className = "choice plan";

    var input = document.createElement("input");
    input.type = "radio";
    input.name = "plan";
    input.value = plan.code;
    if (plan.code === state.catalog.default_plan) input.checked = true;

    var body = document.createElement("span");
    body.className = "choice__body plan__body";

    var features = plan.features.map(function (f) {
      var li = document.createElement("li");
      li.textContent = f;
      return li.outerHTML;
    }).join("");

    body.innerHTML =
      '<span class="choice__name"></span>' +
      '<span><span class="plan__price"></span> <span class="plan__period"></span></span>' +
      '<ul class="plan__features">' + features + "</ul>";
    body.querySelector(".choice__name").textContent = plan.name;
    body.querySelector(".plan__price").textContent = plan.price;
    body.querySelector(".plan__period").textContent = plan.period;

    input.addEventListener("change", function () {
      state.plan = plan;
      refreshPanel();
    });

    label.appendChild(input);
    label.appendChild(body);
    return label;
  }

  function selectPlatform(platform) {
    state.platform = platform;
    setAccent(platform.accent);

    // Choosing a different platform invalidates any product already picked —
    // products don't cross platforms. Clearing it here stops a stale selection
    // from being submitted and rejected by validate_selection on the server.
    state.product = null;

    el.productChoices.innerHTML = "";
    platform.products.forEach(function (product) {
      el.productChoices.appendChild(productCard(product));
    });

    el.productStep.classList.remove("step--locked");
    refreshPanel();
    updateSubmitState();
  }

  function updateSubmitState() {
    el.submit.disabled = !(state.platform && state.product);
  }

  /* ── Issued view ────────────────────────────────────────────────────────── */

  function renderInstall(install) {
    el.installHead.textContent = install.heading;
    el.installSteps.innerHTML = "";

    install.steps.forEach(function (step, i) {
      var row = document.createElement("div");
      row.className = "install__step";

      var num = document.createElement("div");
      num.className = "install__num";
      num.textContent = String(i + 1).padStart(2, "0");

      var body = document.createElement("div");

      var label = document.createElement("div");
      label.className = "install__label";
      label.textContent = step.label;
      body.appendChild(label);

      if (step.detail) {
        var detail = document.createElement("p");
        detail.className = "install__detail";
        detail.textContent = step.detail;
        body.appendChild(detail);
      }

      if (step.code) {
        var code = document.createElement("pre");
        code.className = "install__code";
        code.textContent = step.code;
        body.appendChild(code);
      }

      row.appendChild(num);
      row.appendChild(body);
      el.installSteps.appendChild(row);
    });
  }

  function showIssued(data) {
    state.issued = true;

    el.keyValue.textContent = data.license_key;

    // "Here's the key you already have" is a different message from "here's
    // your new key", and telling someone a key was created when it wasn't
    // invites them to wonder which of the two is live.
    el.issuedTitle.textContent = data.reissued ? "You already have this key" : "Your key is ready";
    el.issuedLede.textContent = data.reissued
      ? data.product.name + " is already licensed for " + data.domain + ". This is that same key — paste it into the module and you're set."
      : data.product.name + " is licensed for " + data.domain + ". Paste this into the module's settings, then run your first sync.";

    setPanelField("platform", data.platform.name);
    setPanelField("product",  data.product.name);
    setPanelField("code",     data.product.code);
    setPanelField("domain",   data.domain);
    setPanelField("plan",     data.plan.name);

    el.licenceExpiry.textContent = "365 days";
    el.licenceExpiry.setAttribute("data-empty", "false");

    el.licence.setAttribute("data-state", "issued");
    el.licenceStamp.textContent = "Issued";

    renderInstall(data.install);

    el.issued.setAttribute("data-visible", "true");
    el.issued.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function resetForAnother() {
    state.issued = false;
    state.product = null;

    el.issued.setAttribute("data-visible", "false");
    el.licence.setAttribute("data-state", "pending");
    el.licenceStamp.textContent = "Not issued";

    // Keep the person's own details — they're requesting a second product for
    // the same store, which is the whole point of per-product keys. Only the
    // product selection is cleared.
    Array.prototype.forEach.call(
      el.form.querySelectorAll('input[name="product_code"]'),
      function (input) { input.checked = false; }
    );

    refreshPanel();
    updateSubmitState();
    document.getElementById("step-product").scrollIntoView({ behavior: "smooth", block: "center" });
  }

  /* ── Submit ─────────────────────────────────────────────────────────────── */

  function onSubmit(event) {
    event.preventDefault();
    el.error.textContent = "";

    if (!state.product) {
      el.error.textContent = "Pick the product you're installing.";
      return;
    }

    var original = el.submit.textContent;
    el.submit.disabled = true;
    el.submit.textContent = "Issuing…";

    fetch("/api/onboarding/signup", {
      method: "POST",
      body: new FormData(el.form),
    })
      .then(function (res) { return res.json(); })
      .then(function (data) {
        if (data && data.success) {
          showIssued(data);
        } else {
          el.error.textContent = (data && data.error) || "We couldn't issue the key. Try again.";
        }
      })
      .catch(function () {
        el.error.textContent = "Couldn't reach the licence server. Check your connection and try again.";
      })
      .then(function () {
        el.submit.disabled = false;
        el.submit.textContent = original;
        updateSubmitState();
      });
  }

  function onCopy() {
    var text = el.keyValue.textContent;
    var done = function () {
      el.copyBtn.setAttribute("data-copied", "true");
      el.copyBtn.textContent = "Copied";
      setTimeout(function () {
        el.copyBtn.setAttribute("data-copied", "false");
        el.copyBtn.textContent = "Copy key";
      }, 2000);
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else {
      fallback();
    }

    // navigator.clipboard is unavailable over plain HTTP, which is exactly how
    // someone testing a staging install will hit this page.
    function fallback() {
      var range = document.createRange();
      range.selectNodeContents(el.keyValue);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      try { document.execCommand("copy"); done(); } catch (err) { /* selection stands; copy by hand */ }
    }
  }

  /* ── Boot ───────────────────────────────────────────────────────────────── */

  function boot() {
    cacheNodes();

    el.form.addEventListener("submit", onSubmit);
    el.storeUrl.addEventListener("input", refreshPanel);
    el.copyBtn.addEventListener("click", onCopy);
    el.anotherBtn.addEventListener("click", resetForAnother);

    fetch("/api/onboarding/catalog")
      .then(function (res) { return res.json(); })
      .then(function (data) {
        state.catalog = data;

        data.platforms.forEach(function (platform) {
          el.platformChoices.appendChild(platformCard(platform));
        });

        data.plans.forEach(function (plan) {
          el.planChoices.appendChild(planCard(plan));
          if (plan.code === data.default_plan) state.plan = plan;
        });

        refreshPanel();
        updateSubmitState();
      })
      .catch(function () {
        el.error.textContent =
          "Couldn't load the product list. Refresh the page, and if it keeps failing let us know.";
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
