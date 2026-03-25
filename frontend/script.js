document.addEventListener("DOMContentLoaded", function () {

  const API_PREDICT = "/predict";   

  let webcamActive  = false;
  let pollInterval  = null;
  let lastBestScore = 0;

  function showUpload() {
    if (webcamActive) stopWebcam();

    document.getElementById("upload-section").style.display  = "block";
    document.getElementById("webcam-section").style.display  = "none";
    document.getElementById("resultCard").style.display      = "none";

    document.getElementById("btn-upload").classList.add("active");
    document.getElementById("btn-webcam").classList.remove("active");
  }

  function showWebcam() {
    document.getElementById("upload-section").style.display  = "none";
    document.getElementById("webcam-section").style.display  = "block";
    document.getElementById("resultCard").style.display      = "none";

    document.getElementById("btn-upload").classList.remove("active");
    document.getElementById("btn-webcam").classList.add("active");

    const feed = document.getElementById("liveFeed");
    feed.src = "/video_feed";
    webcamActive  = true;
    lastBestScore = 0;

    fetch("/reset_result", { method: "POST" });

    document.getElementById("liveResultPanel").style.display = "none";

    pollInterval = setInterval(pollLatestResult, 1500);
  }

  function stopWebcam() {

    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }

    const feed = document.getElementById("liveFeed");
    feed.src = "";
    feed.removeAttribute("src");
    webcamActive  = false;
    lastBestScore = 0;

    fetch("/reset_result", { method: "POST" });
    showUpload();
  }

  window.handleDragOver = function (e) {
    e.preventDefault();
    document.getElementById("dropZone").classList.add("drag-over");
  };

  window.handleDragLeave = function () {
    document.getElementById("dropZone").classList.remove("drag-over");
  };

  window.handleDrop = function (e) {
    e.preventDefault();
    document.getElementById("dropZone").classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      loadPreview(file);
    }
  };

  document.getElementById("fileInput").addEventListener("change", function () {
    if (this.files[0]) loadPreview(this.files[0]);
  });

  function loadPreview(file) {
    const preview    = document.getElementById("previewImage");
    const previewBox = document.getElementById("previewBox");
    const dropZone   = document.getElementById("dropZone");
    const predictBtn = document.getElementById("predictBtn");

    preview.src = URL.createObjectURL(file);
    previewBox.style.display = "block";
    dropZone.style.display   = "none";
    predictBtn.style.display = "block";

    document.getElementById("resultCard").style.display = "none";
  }

  window.clearImage = function () {
    document.getElementById("fileInput").value        = "";
    document.getElementById("previewImage").src       = "";
    document.getElementById("previewBox").style.display  = "none";
    document.getElementById("dropZone").style.display    = "block";
    document.getElementById("predictBtn").style.display  = "none";
    document.getElementById("resultCard").style.display  = "none";
  };

  async function predictImage() {
    const fileInput = document.getElementById("fileInput");
    const file      = fileInput.files[0];

    if (!file) {
      showToast("Please select a leaf image first.");
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_PREDICT, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${response.status}`);
      }

      const data = await response.json();

      if (data.status === "no_leaf") {
        showResult("No leaf detected", 0, false);
      } else {

        showResult(data.disease, data.confidence, true);
      }

    } catch (error) {
      showToast("Prediction failed: " + error.message);
    } finally {
      setLoading(false);
    }
  }

  function showResult(disease, confidencePct, success) {
    const card   = document.getElementById("resultCard");
    const dot    = document.getElementById("resultDot");
    const fill   = document.getElementById("confidenceFill");
    const text   = document.getElementById("confidenceText");
    const value  = document.getElementById("confidenceValue");
    const label  = document.getElementById("disease");

    label.textContent  = disease;
    value.textContent  = confidencePct.toFixed(1) + "%";
    text.textContent   = getConfidenceLabel(confidencePct);

    fill.style.width = "0%";
    fill.className   = "confidence-fill";

    if (!success || confidencePct === 0) {
      fill.classList.add("fill-low");
      dot.classList.add("dot-warn");
    } else if (confidencePct >= 75) {
      fill.classList.add("fill-high");
      dot.classList.add("dot-ok");
    } else if (confidencePct >= 50) {
      fill.classList.add("fill-mid");
      dot.classList.add("dot-mid");
    } else {
      fill.classList.add("fill-low");
      dot.classList.add("dot-warn");
    }

    card.style.display = "block";
   
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fill.style.width = Math.min(confidencePct, 100) + "%";
      });
    });
  }

  function getConfidenceLabel(pct) {
    if (pct >= 85) return "High confidence";
    if (pct >= 65) return "Moderate confidence";
    if (pct >= 40) return "Low confidence";
    return "Very uncertain";
  }

  function setLoading(on) {
    const btn     = document.getElementById("predictBtn");
    const btnText = document.getElementById("predictBtnText");
    const spinner = document.getElementById("predictSpinner");
    btn.disabled          = on;
    btnText.textContent   = on ? "Analysing…" : "Analyse Leaf";
    spinner.style.display = on ? "inline-block" : "none";
  }

  function showToast(msg) {
    let toast = document.getElementById("toast");
    if (!toast) {
      toast = document.createElement("div");
      toast.id = "toast";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.classList.add("toast-show");
    setTimeout(() => toast.classList.remove("toast-show"), 3500);
  }

  async function pollLatestResult() {
    if (!webcamActive) return;
    try {
      const res  = await fetch("/latest_result");
      const data = await res.json();

      if (!data.disease || data.confidence === 0) return;

      if (data.confidence <= lastBestScore) return;
      lastBestScore = data.confidence;

      showLiveResult(data);

      if (data.has_snapshot) {
        const imgRes = await fetch("/snapshot");
        const blob   = await imgRes.blob();
        const url    = URL.createObjectURL(blob);

        const snapshotImg    = document.getElementById("snapshotImg");
        const downloadBtn    = document.getElementById("downloadSnapshot");

        if (snapshotImg.src.startsWith("blob:")) URL.revokeObjectURL(snapshotImg.src);

        snapshotImg.src   = url;
        downloadBtn.href  = url;
      }

    } catch (_) { /* server not ready yet — silently skip */ }
  }

  function showLiveResult(data) {
    const panel = document.getElementById("liveResultPanel");
    const dot   = document.getElementById("liveResultDot");
    const fill  = document.getElementById("liveConfidenceFill");
    const text  = document.getElementById("liveConfidenceText");
    const value = document.getElementById("liveConfidenceValue");
    const label = document.getElementById("liveDisease");

    label.textContent = data.disease.replace(/___/g, " — ").replace(/_/g, " ");
    value.textContent = data.confidence.toFixed(1) + "%";
    text.textContent  = getConfidenceLabel(data.confidence);

    dot.className  = "live-result-dot";
    fill.className = "confidence-fill";
    fill.style.width = "0%";

    if (data.confidence >= 75) {
      fill.classList.add("fill-high"); dot.classList.add("dot-ok");
    } else if (data.confidence >= 50) {
      fill.classList.add("fill-mid");  dot.classList.add("dot-mid");
    } else {
      fill.classList.add("fill-low");  dot.classList.add("dot-warn");
    }

    panel.style.display = "block";

    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = Math.min(data.confidence, 100) + "%";
    }));
  }

  window.resetLiveResult = function () {
    lastBestScore = 0;
    fetch("/reset_result", { method: "POST" });
    document.getElementById("liveResultPanel").style.display = "none";
    document.getElementById("snapshotImg").src = "";
  };

  window.showUpload   = showUpload;
  window.showWebcam   = showWebcam;
  window.stopWebcam   = stopWebcam;
  window.predictImage = predictImage;

});
