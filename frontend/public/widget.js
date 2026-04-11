(function () {
  var HOST = document.currentScript?.getAttribute("data-host") || document.currentScript?.src.replace(/\/widget\.js.*$/, "") || "";

  var STYLES = [
    "#haifa-rag-widget-btn{",
    "  position:fixed;bottom:24px;right:24px;z-index:99999;",
    "  width:56px;height:56px;border-radius:50%;border:none;cursor:pointer;",
    "  background:#1e3a5f;color:#fff;font-size:24px;",
    "  box-shadow:0 4px 12px rgba(0,0,0,0.3);transition:transform .2s;",
    "}",
    "#haifa-rag-widget-btn:hover{transform:scale(1.1);}",
    "#haifa-rag-widget-expand{",
    "  position:fixed;z-index:100000;",
    "  width:32px;height:32px;border-radius:6px;border:none;cursor:pointer;",
    "  background:#1e3a5f;color:#fff;font-size:16px;",
    "  display:none;",
    "}",
    "#haifa-rag-widget-frame{",
    "  position:fixed;bottom:92px;right:24px;z-index:99998;",
    "  width:400px;height:600px;max-height:80vh;max-width:calc(100vw - 48px);",
    "  border:none;border-radius:12px;",
    "  box-shadow:0 8px 32px rgba(0,0,0,0.3);",
    "  display:none;transition:all .3s ease;",
    "}",
    "#haifa-rag-widget-frame.fullscreen{",
    "  top:0;left:0;right:0;bottom:0;",
    "  width:100vw;height:100vh;max-width:100vw;max-height:100vh;",
    "  border-radius:0;",
    "}",
    "@media(max-width:480px){",
    "  #haifa-rag-widget-frame{",
    "    width:calc(100vw - 16px);height:calc(100vh - 120px);",
    "    right:8px;bottom:84px;border-radius:8px;",
    "  }",
    "}",
  ].join("\n");

  var style = document.createElement("style");
  style.textContent = STYLES;
  document.head.appendChild(style);

  var btn = document.createElement("button");
  btn.id = "haifa-rag-widget-btn";
  btn.innerHTML = "&#128172;";
  btn.title = "HAIFA-RAG";
  document.body.appendChild(btn);

  var expandBtn = document.createElement("button");
  expandBtn.id = "haifa-rag-widget-expand";
  expandBtn.innerHTML = "&#x26F6;";
  expandBtn.title = "Full screen";
  document.body.appendChild(expandBtn);

  var iframe = document.createElement("iframe");
  iframe.id = "haifa-rag-widget-frame";
  iframe.src = HOST + "/?embed=1";
  iframe.allow = "clipboard-write";
  document.body.appendChild(iframe);

  var open = false;
  var fullscreen = false;

  function updateExpandPos() {
    if (!open) { expandBtn.style.display = "none"; return; }
    expandBtn.style.display = "block";
    if (fullscreen) {
      expandBtn.style.top = "8px";
      expandBtn.style.right = "8px";
      expandBtn.style.bottom = "auto";
    } else {
      expandBtn.style.bottom = (92 + 600 + 4) + "px";
      expandBtn.style.right = "24px";
      expandBtn.style.top = "auto";
    }
  }

  btn.addEventListener("click", function () {
    if (fullscreen) {
      fullscreen = false;
      iframe.classList.remove("fullscreen");
      updateExpandPos();
      return;
    }
    open = !open;
    iframe.style.display = open ? "block" : "none";
    btn.innerHTML = open ? "&#10005;" : "&#128172;";
    updateExpandPos();
  });

  expandBtn.addEventListener("click", function () {
    fullscreen = !fullscreen;
    iframe.classList.toggle("fullscreen", fullscreen);
    expandBtn.innerHTML = fullscreen ? "&#x2716;" : "&#x26F6;";
    expandBtn.title = fullscreen ? "Exit full screen" : "Full screen";
    btn.style.display = fullscreen ? "none" : "block";
    updateExpandPos();
  });
})();
