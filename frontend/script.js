document.addEventListener("DOMContentLoaded", function () {

  const API_PREDICT = "/predict";

  let webcamActive  = false;
  let pollInterval  = null;
  let lastBestScore = 0;

  const HEALTHY_KEYWORDS = ["healthy"];

  function isHealthy(label) {
    return HEALTHY_KEYWORDS.some(k => label.toLowerCase().includes(k));
  }

  function cleanLabel(raw) {
    
    let prefix = "";
    if (raw.startsWith("Uncertain (") && raw.endsWith(")")) {
      prefix = "Uncertain — ";
      raw = raw.slice(11, -1);
    }

    let parts = raw.split("___");
    if (parts.length < 2) {
     
      return prefix + raw.replace(/_/g, " ").replace(/\s+/g, " ").trim();
    }

    let crop = parts[0].replace(/\s*\(.*?\)/g, "").replace(/_/g, " ").trim();

    let disease = parts[1].replace(/_+$/g, "").replace(/_/g, " ").trim();

    let cropLower    = crop.toLowerCase();
    let diseaseLower = disease.toLowerCase();
    if (diseaseLower.startsWith(cropLower)) {
      disease = disease.slice(crop.length).trim();
    }

    disease = disease.replace(/\w/g, c => c.toUpperCase());

    if (!disease || disease.toLowerCase() === "healthy") {
      return prefix + crop + " — Healthy";
    }

    return prefix + crop + " — " + disease;
  }

  function setPanel(name) {
    document.getElementById("rightPanel").dataset.panel = name;
  }

  function showUpload() {
    if (webcamActive) stopWebcam();
    document.getElementById("upload-section").style.display = "block";
    document.getElementById("webcam-section").style.display = "none";
    document.getElementById("btn-upload").classList.add("active");
    document.getElementById("btn-webcam").classList.remove("active");
    resetUploadResult();
    setPanel("upload");
  }

  function showWebcam() {
    document.getElementById("upload-section").style.display = "none";
    document.getElementById("webcam-section").style.display = "block";
    document.getElementById("btn-upload").classList.remove("active");
    document.getElementById("btn-webcam").classList.add("active");
    setPanel("webcam-wait");

    const feed = document.getElementById("liveFeed");
    feed.src = "/video_feed";
    webcamActive  = true;
    lastBestScore = 0;

    fetch("/reset_result", { method: "POST" });
    if (pollInterval) clearInterval(pollInterval);
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

    document.getElementById("upload-section").style.display = "block";
    document.getElementById("webcam-section").style.display = "none";
    document.getElementById("btn-upload").classList.add("active");
    document.getElementById("btn-webcam").classList.remove("active");
    setPanel("upload");
  }

  function resetUploadResult() {
    document.getElementById("resultPlaceholder").style.display = "flex";
    document.getElementById("resultContent").style.display     = "none";
    const card = document.getElementById("resultCard");
    card.className = "result-card state-empty panel-upload";
  }

  function showPlaceholder() {
    document.getElementById("resultPlaceholder").style.display = "flex";
    document.getElementById("resultContent").style.display     = "none";
    const card = document.getElementById("resultCard");
    card.className = "result-card state-empty panel-upload";
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
    if (file && file.type.startsWith("image/")) loadPreview(file);
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
    showPlaceholder();
  }

  window.clearImage = function () {
    document.getElementById("fileInput").value              = "";
    document.getElementById("previewImage").src             = "";
    document.getElementById("previewBox").style.display     = "none";
    document.getElementById("dropZone").style.display       = "block";
    document.getElementById("predictBtn").style.display     = "none";
    showPlaceholder();
  };

  async function predictImage() {
    const fileInput = document.getElementById("fileInput");
    const file      = fileInput.files[0];
    if (!file) { showToast("Please select a leaf image first."); return; }

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(API_PREDICT, { method: "POST", body: formData });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${response.status}`);
      }
      const data = await response.json();
      if (data.status === "no_leaf") {
        showUploadResult("No leaf detected", 0, false);
      } else {
        showUploadResult(data.disease, data.confidence, true);
      }
    } catch (error) {
      showToast("Prediction failed: " + error.message);
    } finally {
      setLoading(false);
    }
  }

  function showUploadResult(disease, confidencePct, success) {
    const card        = document.getElementById("resultCard");
    const placeholder = document.getElementById("resultPlaceholder");
    const content     = document.getElementById("resultContent");
    const badge       = document.getElementById("resultBadge");
    const badgeText   = document.getElementById("resultBadgeText");
    const dot         = document.getElementById("resultDot");
    const label       = document.getElementById("disease");
    const fill        = document.getElementById("confidenceFill");
    const value       = document.getElementById("confidenceValue");
    const status      = document.getElementById("confidenceText");

    placeholder.style.display = "none";
    content.style.display     = "block";

    label.textContent = cleanLabel(disease);
    value.textContent = confidencePct.toFixed(1) + "%";
    status.textContent = getConfidenceLabel(confidencePct);

    fill.style.width = "0%";
    fill.className   = "confidence-fill";
    dot.className    = "badge-dot";
    badge.className  = "result-badge";

    const healthy = success && isHealthy(disease);

    if (!success || confidencePct === 0) {
      
      card.className     = "result-card state-noleaf panel-upload";
      badge.classList.add("badge-warn");
      dot.classList.add("dot-warn");
      badgeText.textContent = "No Leaf Detected";
      fill.classList.add("fill-low");
    } else if (healthy) {
   
      card.className     = "result-card state-healthy panel-upload";
      badge.classList.add("badge-healthy");
      dot.classList.add("dot-healthy");
      badgeText.textContent = "Healthy Plant ✓";
      fill.classList.add("fill-healthy");
    } else {
      
      card.className     = "result-card state-disease panel-upload";
      badge.classList.add("badge-disease");
      dot.classList.add("dot-disease");
      badgeText.textContent = "Disease Detected";
      fill.classList.add("fill-disease");
    }

    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = Math.min(confidencePct, 100) + "%";
    }));
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
        const snapshotImg = document.getElementById("snapshotImg");
        const downloadBtn = document.getElementById("downloadSnapshot");
        if (snapshotImg.src.startsWith("blob:")) URL.revokeObjectURL(snapshotImg.src);
        snapshotImg.src  = url;
        downloadBtn.href = url;
      }
    } catch (_) {}
  }

  function showLiveResult(data) {
    const panel     = document.getElementById("liveResultPanel");
    const dot       = document.getElementById("liveResultDot");
    const fill      = document.getElementById("liveConfidenceFill");
    const text      = document.getElementById("liveConfidenceText");
    const value     = document.getElementById("liveConfidenceValue");
    const label     = document.getElementById("liveDisease");
    const badge     = document.getElementById("liveBadge");
    const badgeDot  = document.getElementById("liveBadgeDot");
    const badgeText = document.getElementById("liveBadgeText");

    const disease = cleanLabel(data.disease);
    label.textContent = disease;
    value.textContent = data.confidence.toFixed(1) + "%";
    text.textContent  = getConfidenceLabel(data.confidence);

    fill.style.width = "0%";
    fill.className   = "confidence-fill";
    dot.className    = "live-result-dot";
    badge.className  = "result-badge";
    badgeDot.className = "badge-dot";

    const healthy = isHealthy(data.disease);

    if (healthy) {
      fill.classList.add("fill-healthy");
      dot.classList.add("dot-healthy");
      badge.classList.add("badge-healthy");
      badgeDot.classList.add("dot-healthy");
      badgeText.textContent = "Healthy Plant ✓";
      panel.className = "result-card live-result-card live-state-healthy panel-webcam-result";
    } else {
      fill.classList.add("fill-disease");
      dot.classList.add("dot-disease");
      badge.classList.add("badge-disease");
      badgeDot.classList.add("dot-disease");
      badgeText.textContent = "Disease Detected";
      panel.className = "result-card live-result-card live-state-disease panel-webcam-result";
    }

    setPanel("webcam-result");
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.width = Math.min(data.confidence, 100) + "%";
    }));
  }

  window.resetLiveResult = function () {
    lastBestScore = 0;
    fetch("/reset_result", { method: "POST" });
    document.getElementById("liveResultPanel").className = "result-card live-result-card panel-webcam-result";
    setPanel("webcam-wait");
    document.getElementById("snapshotImg").src = "";
  };

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

  window.showUpload   = showUpload;
  window.showWebcam   = showWebcam;
  window.stopWebcam   = stopWebcam;
  window.predictImage = predictImage;
});
