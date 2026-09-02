"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import { Aperture, Cpu, Crosshair, ImageIcon, MapPin, Maximize2, RotateCcw, ScanLine, UploadCloud, ZoomIn } from "lucide-react";
import Navigation from "@/components/Navigation";
import SpaceScene from "@/components/SpaceScene";
import { getHealth, getModelInfo, resolveApiUrl, runSuperResolution } from "@/services/api";
import type { HealthResponse, LocalImageMeta, ModelInfoResponse, SuperResolutionResult } from "@/types/api";

const processingStages = [
  "DATA ACQUISITION",
  "PREPROCESSING",
  "FEATURE EXTRACTION",
  "AI INFERENCE",
  "IMAGE RECONSTRUCTION",
];

const pipelineStages = [
  "SATELLITE DATA",
  "PREPROCESSING",
  "FEATURE EXTRACTION",
  "GEOSR NEURAL ENGINE",
  "SUPER-RESOLUTION",
  "RECONSTRUCTION",
  "HIGH-RESOLUTION OUTPUT",
];

const timeline = [
  "MISSION INITIALIZED",
  "SATELLITE DATA RECEIVED",
  "IMAGE PREPROCESSING",
  "AI INFERENCE",
  "SUPER-RESOLUTION COMPLETE",
  "OUTPUT GENERATED",
  "GEOSPATIAL ANALYSIS READY",
];

const satelliteFleet = [
  { id: "GS-01", orbit: "LEO", altitude: "542 KM", velocity: "7.61 KM/S", mission: "EO IMAGING", signal: "98%" },
  { id: "GS-02", orbit: "MEO", altitude: "20120 KM", velocity: "3.88 KM/S", mission: "RELAY", signal: "94%" },
  { id: "GS-CUBE", orbit: "LEO", altitude: "488 KM", velocity: "7.68 KM/S", mission: "AOI SCAN", signal: "91%" },
];

function formatBytes(bytes: number): string {
  if (bytes === 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function resultImageSrc(result: SuperResolutionResult | null): string | null {
  return result?.output.sr_image_b64 ? `data:image/png;base64,${result.output.sr_image_b64}` : null;
}

function uncertaintyImageSrc(result: SuperResolutionResult | null): string | null {
  return result?.output.uncertainty_map_b64 ? `data:image/png;base64,${result.output.uncertainty_map_b64}` : null;
}

async function createLocalMeta(file: File): Promise<LocalImageMeta> {
  const extension = file.name.split(".").pop()?.toUpperCase() || "UNKNOWN";
  const acquisitionTime = new Date(file.lastModified).toLocaleString();
  const previewable = file.type.startsWith("image/") && !/tiff?/i.test(file.type) && !/\.tiff?$/i.test(file.name);

  if (!previewable) {
    return {
      filename: file.name,
      width: null,
      height: null,
      sizeBytes: file.size,
      format: extension,
      acquisitionTime,
      previewUrl: null,
    };
  }

  const previewUrl = URL.createObjectURL(file);
  const size = await new Promise<{ width: number; height: number }>((resolve) => {
    const image = new window.Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => resolve({ width: 0, height: 0 });
    image.src = previewUrl;
  });

  return {
    filename: file.name,
    width: size.width || null,
    height: size.height || null,
    sizeBytes: file.size,
    format: extension,
    acquisitionTime,
    previewUrl,
  };
}

export default function MissionApp() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [localMeta, setLocalMeta] = useState<LocalImageMeta | null>(null);
  const [result, setResult] = useState<SuperResolutionResult | null>(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [comparison, setComparison] = useState(50);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const referenceInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadSystemState() {
      try {
        const [healthResponse, modelResponse] = await Promise.allSettled([getHealth(), getModelInfo()]);
        if (cancelled) {
          return;
        }

        if (healthResponse.status === "fulfilled") {
          setHealth(healthResponse.value);
        } else {
          setApiError(healthResponse.reason instanceof Error ? healthResponse.reason.message : "API unavailable");
        }

        if (modelResponse.status === "fulfilled") {
          setModelInfo(modelResponse.value);
        }
      } catch (loadError) {
        if (!cancelled) {
          setApiError(loadError instanceof Error ? loadError.message : "API unavailable");
        }
      }
    }

    loadSystemState();
    const interval = window.setInterval(loadSystemState, 20000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!processing) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      setProgress((current) => Math.min(94, current + Math.max(1, Math.round((100 - current) * 0.08))));
    }, 560);

    return () => window.clearInterval(timer);
  }, [processing]);

  useEffect(() => {
    return () => {
      if (localMeta?.previewUrl) {
        URL.revokeObjectURL(localMeta.previewUrl);
      }
    };
  }, [localMeta?.previewUrl]);

  const online = health?.status === "ok" || health?.model_ready === true;
  const scaleFactor = result?.model.scale_factor ?? modelInfo?.scale_factor ?? null;
  const outputImage = resultImageSrc(result);
  const uncertaintyImage = uncertaintyImageSrc(result);
  const activeStage = Math.min(processingStages.length - 1, Math.floor((progress / 100) * processingStages.length));

  const dashboardCards = useMemo(
    () => [
      {
        label: "INPUT IMAGERY",
        value: result?.input.width && result.input.height ? `${result.input.width} x ${result.input.height}` : localMeta?.width && localMeta.height ? `${localMeta.width} x ${localMeta.height}` : "No image",
        sub: result?.input.format ?? localMeta?.format ?? "Awaiting upload",
      },
      {
        label: "OUTPUT RESOLUTION",
        value: result?.output.width && result.output.height ? `${result.output.width} x ${result.output.height}` : "Awaiting inference",
        sub: result?.output.bands ? `${result.output.bands} bands` : "No backend result yet",
      },
      {
        label: "ENHANCEMENT",
        value: scaleFactor ? `${scaleFactor}x` : "Awaiting model",
        sub: modelInfo?.input_resolution ?? result?.input.resolution ?? "Model metadata",
      },
      {
        label: "PROCESSING TIME",
        value: result?.output.inference_time_s ? `${result.output.inference_time_s}s` : processing ? `${progress}%` : "Idle",
        sub: result?.output.tiled_inference ? "Tiled inference" : "Direct inference",
      },
      {
        label: "MODEL STATUS",
        value: modelInfo?.architecture ?? health?.status?.toUpperCase() ?? "Unknown",
        sub: modelInfo?.checkpoint_loaded || health?.checkpoint_loaded ? "Checkpoint loaded" : "Checkpoint not reported",
      },
      {
        label: "GPU STATUS",
        value: health?.cuda_available || modelInfo?.cuda_available ? "CUDA READY" : health ? "CPU MODE" : "Unknown",
        sub: health?.device ?? modelInfo?.device ?? "Device telemetry",
      },
    ],
    [health, localMeta, modelInfo, processing, progress, result, scaleFactor],
  );

  async function handleFile(file: File | null) {
    if (!file) {
      return;
    }
    setError(null);
    setResult(null);
    setSelectedFile(file);
    const nextMeta = await createLocalMeta(file);
    setLocalMeta((previous) => {
      if (previous?.previewUrl) {
        URL.revokeObjectURL(previous.previewUrl);
      }
      return nextMeta;
    });
  }

  async function handleLoadDemo(url: string, filename: string) {
    try {
      setError(null);
      const res = await fetch(url);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: blob.type || "image/tiff" });
      await handleFile(file);
    } catch {
      setError("Could not load demo satellite image");
    }
  }

  useEffect(() => {
    // Automatically load the preloaded Sentinel-2 sample on first visit
    handleLoadDemo("/sample_sentinel2.tif", "Sentinel-2_Sample_AOI.tif");
  }, []);


  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    handleFile(event.dataTransfer.files?.[0] ?? null);
  }


  async function runInference() {
    if (!selectedFile) {
      setError("Select a satellite image before launching inference.");
      return;
    }

    setProcessing(true);
    setProgress(7);
    setError(null);

    try {
      const response = await runSuperResolution(selectedFile, referenceFile);
      setResult(response);
      setProgress(100);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Inference failed.");
    } finally {
      setProcessing(false);
    }
  }

  function openDownload(path?: string) {
    if (!path) {
      return;
    }
    window.open(resolveApiUrl(path), "_blank", "noopener,noreferrer");
  }

  return (
    <main className="min-h-screen bg-radial-space text-slate-50">
      <Navigation online={online} />
      <Hero online={online} modelInfo={modelInfo} health={health} onLaunch={() => document.getElementById("mission")?.scrollIntoView({ behavior: "smooth" })} />

      <section id="mission" className="relative overflow-hidden border-y border-cyan-signal/10 bg-space-950 py-16 sm:py-20">
        <div className="mission-grid absolute inset-0 bg-mission-grid opacity-50" aria-hidden="true" />
        <div className="noise-overlay absolute inset-0" aria-hidden="true" />
        <div className="relative mx-auto max-w-[1480px] px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="font-telemetry mb-4 flex items-center gap-2 text-xs uppercase tracking-[0.22em] text-cyan-signal">
                <span className={online ? "status-dot" : "warning-dot"} />
                {online ? "NEURAL ENGINE ONLINE" : "NEURAL ENGINE STANDBY"}
              </div>
              <h2 className="font-display text-3xl font-semibold uppercase tracking-[0.12em] text-white sm:text-5xl">GEOSR MISSION CONTROL</h2>
              <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">Earth Observation Super-Resolution System connected to the existing GeoSR-AI FastAPI inference pipeline.</p>
            </div>
            <ApiStatusPanel health={health} modelInfo={modelInfo} apiError={apiError} />
          </div>

          <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-6">
            {dashboardCards.map((card) => (
              <StatusCard key={card.label} {...card} />
            ))}
          </div>
        </div>
      </section>

      <section id="processing" className="relative bg-space-900 py-16 sm:py-20">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(0,217,255,.12),transparent_28%),radial-gradient(circle_at_82%_60%,rgba(108,99,255,.12),transparent_32%)]" aria-hidden="true" />
        <div className="relative mx-auto max-w-[1480px] px-4 sm:px-6 lg:px-8">
          <SectionHeader eyebrow="SATELLITE IMAGE PROCESSING CENTER" title="SUPER-RESOLUTION WORKSPACE" copy="Upload a real satellite image, launch the existing backend inference endpoint, then inspect the returned preview, metadata, uncertainty, downloads, and metrics." />

          <div className="mt-10 grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px_minmax(0,1fr)]">
            <InputPanel
              localMeta={localMeta}
              selectedFile={selectedFile}
              referenceFile={referenceFile}
              onDrop={onDrop}
              onSelect={() => fileInputRef.current?.click()}
              onReferenceSelect={() => referenceInputRef.current?.click()}
              onLoadDemo={handleLoadDemo}
              onRun={runInference}
              processing={processing}
            />

            <ProcessingCore progress={progress} activeStage={activeStage} processing={processing} error={error} />

            <OutputPanel result={result} outputImage={outputImage} uncertaintyImage={uncertaintyImage} onDownload={openDownload} />
          </div>

          <input
            ref={fileInputRef}
            className="hidden"
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
            onChange={(event: ChangeEvent<HTMLInputElement>) => handleFile(event.target.files?.[0] ?? null)}
          />
          <input
            ref={referenceInputRef}
            className="hidden"
            type="file"
            accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
            onChange={(event: ChangeEvent<HTMLInputElement>) => setReferenceFile(event.target.files?.[0] ?? null)}
          />
        </div>
      </section>

      <ImageComparisonSection
        inputImage={localMeta?.previewUrl ?? null}
        outputImage={outputImage}
        comparison={comparison}
        setComparison={setComparison}
        scaleFactor={scaleFactor}
      />

      <SatelliteIntelligenceSection />
      <MapSection result={result} scaleFactor={scaleFactor} />
      <AnalyticsSection result={result} modelInfo={modelInfo} />
      <TechnologySection />
    </main>
  );
}

function Hero({
  online,
  modelInfo,
  health,
  onLaunch,
}: {
  online: boolean;
  modelInfo: ModelInfoResponse | null;
  health: HealthResponse | null;
  onLaunch: () => void;
}) {
  return (
    <section id="hero" className="relative min-h-screen overflow-hidden bg-space-950">
      <SpaceScene className="left-[18%] top-0 h-full w-[120%] md:left-[28%] md:w-[92%]" />
      <div className="hero-vignette absolute inset-0" aria-hidden="true" />
      <div className="absolute inset-x-0 bottom-0 h-28 bg-gradient-to-t from-space-950 to-transparent" aria-hidden="true" />
      <div className="relative z-10 mx-auto flex min-h-screen max-w-[1480px] items-center px-4 pb-20 pt-28 sm:px-6 lg:px-8">
        <div className="max-w-2xl">
          <div className="font-telemetry mb-5 flex items-center gap-3 text-xs uppercase tracking-[0.24em] text-cyan-signal">
            <span className={online ? "status-dot" : "warning-dot"} />
            EARTH OBSERVATION SYSTEM {online ? "ONLINE" : "INITIALIZING"}
          </div>
          <h1 className="font-display text-6xl font-semibold uppercase tracking-[0.08em] text-white sm:text-7xl lg:text-8xl">GeoSR AI</h1>
          <p className="font-display mt-4 text-2xl font-semibold uppercase tracking-[0.18em] text-cyan-signal sm:text-4xl">SEE MORE. RESOLVE BETTER.</p>
          <p className="mt-6 max-w-xl text-base leading-8 text-slate-300 sm:text-lg">Transform low-resolution satellite imagery into high-resolution geospatial intelligence using deep learning.</p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row">
            <button
              onClick={onLaunch}
              className="light-sweep border border-cyan-signal/60 bg-cyan-signal px-6 py-4 font-telemetry text-xs font-semibold uppercase tracking-[0.22em] text-space-950 shadow-glow transition hover:translate-y-[-1px]"
            >
              LAUNCH GEOSR
            </button>
            <a
              href="#technology"
              className="border border-white/15 bg-white/5 px-6 py-4 text-center font-telemetry text-xs font-semibold uppercase tracking-[0.22em] text-white transition hover:border-cyan-signal/50 hover:text-cyan-signal"
            >
              EXPLORE TECHNOLOGY
            </a>
          </div>

          <div className="mt-10 grid max-w-2xl grid-cols-1 gap-3 sm:grid-cols-3">
            <MiniTelemetry label="MISSION STATUS" value={online ? "ONLINE" : "STANDBY"} active={online} />
            <MiniTelemetry label="MODEL" value={modelInfo?.model_name ?? "GeoSR Neural Engine"} />
            <MiniTelemetry label="DATA" value="EARTH OBSERVATION" />
          </div>
        </div>

        <div className="pointer-events-none absolute right-4 top-[22%] hidden w-72 lg:block">
          <div className="hud-panel p-5">
            <p className="font-telemetry text-[10px] uppercase tracking-[0.24em] text-cyan-signal">SATELLITE</p>
            <div className="mt-4 grid grid-cols-2 gap-4 font-telemetry text-xs uppercase">
              <TelemetryPair label="ID" value="GS-01" />
              <TelemetryPair label="ORBIT" value="LEO" />
              <TelemetryPair label="ALTITUDE" value="542 KM" />
              <TelemetryPair label="VELOCITY" value="7.61 KM/S" />
              <TelemetryPair label="SIGNAL" value={online ? "98%" : "LINKING"} />
              <TelemetryPair label="IMAGING" value={online ? "ACTIVE" : "STANDBY"} />
            </div>
          </div>
        </div>

        <div className="pointer-events-none absolute bottom-[12%] right-[7%] hidden w-60 lg:block">
          <div className="hud-panel p-4">
            <p className="font-telemetry text-[10px] uppercase tracking-[0.24em] text-cyan-signal">SCANNING AOI</p>
            <div className="mt-3 grid gap-2 font-telemetry text-xs uppercase">
              <TelemetryPair label="LAT" value="21.2514 DEG" />
              <TelemetryPair label="LON" value="81.6296 DEG" />
              <TelemetryPair label="RESOLUTION" value={modelInfo?.input_resolution ?? "10m"} />
              <TelemetryPair label="STATUS" value={health?.model_ready ? "ACQUIRING DATA" : "AWAITING MODEL"} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function ApiStatusPanel({ health, modelInfo, apiError }: { health: HealthResponse | null; modelInfo: ModelInfoResponse | null; apiError: string | null }) {
  return (
    <div className="hud-panel w-full max-w-xl p-5">
      <div className="grid gap-3 sm:grid-cols-3">
        <TelemetryPair label="API" value={apiError ? "UNAVAILABLE" : health?.status?.toUpperCase() ?? "CHECKING"} />
        <TelemetryPair label="DEVICE" value={health?.device ?? modelInfo?.device ?? "UNKNOWN"} />
        <TelemetryPair label="VERSION" value={health?.version ?? "1.0.0"} />
      </div>
      {apiError ? <p className="mt-4 text-xs leading-5 text-amber-200">Backend status: {apiError}</p> : null}
    </div>
  );
}

function StatusCard({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="hud-panel p-4 transition duration-300 hover:-translate-y-1 hover:border-cyan-signal/40">
      <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-muted">{label}</p>
      <p className="font-display mt-4 min-h-8 text-xl font-semibold uppercase tracking-[0.08em] text-white">{value}</p>
      <p className="font-telemetry mt-3 text-[11px] uppercase tracking-[0.16em] text-cyan-signal/80">{sub}</p>
    </div>
  );
}

function InputPanel({
  localMeta,
  selectedFile,
  referenceFile,
  onDrop,
  onSelect,
  onReferenceSelect,
  onLoadDemo,
  onRun,
  processing,
}: {
  localMeta: LocalImageMeta | null;
  selectedFile: File | null;
  referenceFile: File | null;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onSelect: () => void;
  onReferenceSelect: () => void;
  onLoadDemo: (url: string, filename: string) => void;
  onRun: () => void;
  processing: boolean;
}) {
  return (
    <div className="hud-panel p-5">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-cyan-signal">INPUT SATELLITE IMAGE</p>
          <h3 className="font-display mt-2 text-xl font-semibold uppercase tracking-[0.1em] text-white">DATA ACQUISITION</h3>
        </div>
        <UploadCloud className="h-6 w-6 text-cyan-signal" aria-hidden="true" />
      </div>

      <div
        className="image-shell relative flex items-center justify-center overflow-hidden border border-dashed border-cyan-signal/40 bg-black/25"
        onDragOver={(event) => event.preventDefault()}
        onDrop={onDrop}
      >
        {localMeta?.previewUrl ? (
          <img src={localMeta.previewUrl} alt="Input satellite preview" className="h-full w-full object-cover" />
        ) : (
          <div className="px-6 text-center">
            <ImageIcon className="mx-auto h-12 w-12 text-cyan-signal/80" aria-hidden="true" />
            <p className="font-display mt-5 text-xl font-semibold uppercase tracking-[0.12em] text-white">Drag & Drop Satellite Image</p>
            <p className="mt-2 text-sm text-muted">PNG, JPG, TIFF, or GeoTIFF</p>
            {selectedFile ? <p className="font-telemetry mt-4 text-xs uppercase tracking-[0.16em] text-cyan-signal">{selectedFile.name}</p> : null}
          </div>
        )}
        <div className="scanline absolute inset-y-0 w-1/2 opacity-40" aria-hidden="true" />
      </div>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <button onClick={onSelect} className="border border-cyan-signal/40 bg-cyan-signal/10 px-4 py-3 font-telemetry text-xs uppercase tracking-[0.18em] text-cyan-signal transition hover:bg-cyan-signal/20">
          SELECT IMAGE
        </button>
        <button onClick={onRun} disabled={processing} className="light-sweep border border-cyan-signal bg-cyan-signal px-4 py-3 font-telemetry text-xs font-semibold uppercase tracking-[0.18em] text-space-950 disabled:cursor-wait disabled:opacity-70">
          {processing ? "PROCESSING" : "RUN GEOSR"}
        </button>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <button
          onClick={() => onLoadDemo("/sample_sentinel2.tif", "Sentinel-2_Sample_AOI.tif")}
          className="border border-white/15 bg-white/[0.04] px-2 py-2 text-center font-telemetry text-[10px] uppercase tracking-[0.14em] text-cyan-signal transition hover:border-cyan-signal/60"
        >
          🛰️ SAMPLE SENTINEL-2
        </button>
        <button
          onClick={() => onLoadDemo("/sample_preview.png", "Satellite_RGB_Sample.png")}
          className="border border-white/15 bg-white/[0.04] px-2 py-2 text-center font-telemetry text-[10px] uppercase tracking-[0.14em] text-slate-300 transition hover:border-cyan-signal/60"
        >
          🏙️ SAMPLE RGB SCENE
        </button>
      </div>

      <button onClick={onReferenceSelect} className="mt-3 w-full border border-white/10 bg-white/[0.03] px-4 py-3 font-telemetry text-[11px] uppercase tracking-[0.16em] text-slate-300 transition hover:border-cyan-signal/40 hover:text-cyan-signal">
        {referenceFile ? `REFERENCE: ${referenceFile.name}` : "OPTIONAL HR REFERENCE FOR METRICS"}
      </button>


      <div className="mt-6">
        <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-muted">IMAGE METADATA</p>
        <div className="mt-3 grid gap-3 font-telemetry text-xs uppercase">
          <TelemetryPair label="Resolution" value={localMeta?.width && localMeta.height ? `${localMeta.width} x ${localMeta.height}` : "Awaiting backend parse"} />
          <TelemetryPair label="Format" value={localMeta?.format ?? "No file"} />
          <TelemetryPair label="Size" value={localMeta ? formatBytes(localMeta.sizeBytes) : "No file"} />
          <TelemetryPair label="Coordinates" value="Parsed when geospatial metadata is returned" />
          <TelemetryPair label="Acquisition Time" value={localMeta?.acquisitionTime ?? "No image selected"} />
        </div>
      </div>
    </div>
  );
}

function ProcessingCore({ progress, activeStage, processing, error }: { progress: number; activeStage: number; processing: boolean; error: string | null }) {
  return (
    <div className="hud-panel flex min-h-[620px] flex-col justify-between p-5">
      <div>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-cyan-signal">AI PROCESSING ENGINE</p>
            <h3 className="font-display mt-2 text-lg font-semibold uppercase tracking-[0.12em] text-white">GEOSR NEURAL ENGINE</h3>
          </div>
          <Cpu className="h-6 w-6 text-cyan-signal" aria-hidden="true" />
        </div>

        <div className="relative mx-auto mt-8 flex h-40 w-40 items-center justify-center rounded-full border border-cyan-signal/25 bg-cyan-signal/5">
          <div className="absolute inset-4 rounded-full border border-orbit-purple/30" style={{ animation: "pulse-soft 3s ease-in-out infinite" }} />
          <div className="absolute inset-8 rounded-full border border-cyan-signal/30" />
          <BrainNetwork />
          <span className="font-telemetry text-2xl font-semibold text-cyan-signal">{processing ? `${progress}%` : "IDLE"}</span>
        </div>

        <div className="mt-8 grid gap-3">
          {pipelineStages.map((stage, index) => (
            <div key={stage} className="relative flex items-center gap-3">
              <span className={`h-2 w-2 rounded-full ${processing && index <= activeStage ? "bg-cyan-signal shadow-[0_0_14px_rgba(0,217,255,.9)]" : "bg-white/20"}`} />
              <span className={`font-telemetry text-[11px] uppercase tracking-[0.18em] ${processing && index <= activeStage ? "text-cyan-signal" : "text-slate-400"}`}>{stage}</span>
              {index < pipelineStages.length - 1 ? <span className="absolute left-[3px] top-5 h-3 w-px bg-cyan-signal/20" /> : null}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-8">
        <div className="h-2 overflow-hidden bg-white/10">
          <div className="h-full bg-gradient-to-r from-cyan-signal to-orbit-purple transition-all duration-500" style={{ width: `${progress}%` }} />
        </div>
        <div className="mt-4 grid gap-2 font-telemetry text-[11px] uppercase tracking-[0.16em]">
          {processingStages.map((stage, index) => (
            <div key={stage} className="flex justify-between text-slate-300">
              <span>{`${String(index + 1).padStart(2, "0")} ${stage}`}</span>
              <span className={index < activeStage ? "text-orbit-green" : index === activeStage && processing ? "text-cyan-signal" : "text-slate-500"}>
                {index < activeStage ? "DONE" : index === activeStage && processing ? "ACTIVE" : "WAIT"}
              </span>
            </div>
          ))}
        </div>
        {error ? <p className="mt-4 border border-red-400/25 bg-red-500/10 p-3 text-sm leading-6 text-red-100">{error}</p> : null}
      </div>
    </div>
  );
}

function OutputPanel({ result, outputImage, uncertaintyImage, onDownload }: { result: SuperResolutionResult | null; outputImage: string | null; uncertaintyImage: string | null; onDownload: (path?: string) => void }) {
  return (
    <div className="hud-panel p-5">
      <div className="mb-5 flex items-center justify-between gap-3">
        <div>
          <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-cyan-signal">SUPER-RESOLUTION OUTPUT</p>
          <h3 className="font-display mt-2 text-xl font-semibold uppercase tracking-[0.1em] text-white">HIGH-RES VIEWER</h3>
        </div>
        <Aperture className="h-6 w-6 text-cyan-signal" aria-hidden="true" />
      </div>

      <div className="image-shell relative flex items-center justify-center overflow-hidden border border-cyan-signal/20 bg-black/30">
        {outputImage ? (
          <img src={outputImage} alt="GeoSR super-resolution output" className="h-full w-full object-cover" />
        ) : (
          <div className="px-6 text-center">
            <ScanLine className="mx-auto h-12 w-12 text-cyan-signal/80" aria-hidden="true" />
            <p className="font-display mt-5 text-xl font-semibold uppercase tracking-[0.12em] text-white">No output yet</p>
            <p className="mt-2 text-sm text-muted">Run inference to display backend-generated super-resolution imagery.</p>
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-4 gap-2">
        {[
          { icon: ZoomIn, label: "Zoom" },
          { icon: Maximize2, label: "Fullscreen" },
          { icon: RotateCcw, label: "Reset" },
          { icon: Crosshair, label: "Center" },
        ].map((item) => (
          <button key={item.label} className="flex h-10 items-center justify-center border border-white/10 bg-white/[0.03] text-slate-300 transition hover:border-cyan-signal/40 hover:text-cyan-signal" title={item.label} aria-label={item.label}>
            <item.icon className="h-4 w-4" aria-hidden="true" />
          </button>
        ))}
      </div>

      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        <a href="#comparison" className="border border-white/10 bg-white/[0.03] px-3 py-3 text-center font-telemetry text-[11px] uppercase tracking-[0.14em] text-slate-300 transition hover:border-cyan-signal/40 hover:text-cyan-signal">
          COMPARE
        </a>
        <button onClick={() => onDownload(result?.downloads.sr_geotiff)} disabled={!result} className="border border-cyan-signal/40 bg-cyan-signal/10 px-3 py-3 font-telemetry text-[11px] uppercase tracking-[0.14em] text-cyan-signal disabled:cursor-not-allowed disabled:opacity-50">
          DOWNLOAD RESULT
        </button>
        <button onClick={() => onDownload(result?.downloads.uncertainty)} disabled={!result} className="border border-white/10 bg-white/[0.03] px-3 py-3 font-telemetry text-[11px] uppercase tracking-[0.14em] text-slate-300 disabled:cursor-not-allowed disabled:opacity-45">
          UNCERTAINTY
        </button>
      </div>

      <div className="mt-6 grid gap-3 font-telemetry text-xs uppercase">
        <TelemetryPair label="Output" value={result?.output.width && result.output.height ? `${result.output.width} x ${result.output.height}` : "Awaiting inference"} />
        <TelemetryPair label="Inference" value={result?.output.inference_time_s ? `${result.output.inference_time_s}s` : "No backend result"} />
        <TelemetryPair label="Download" value={result ? "READY" : "WAITING"} />
      </div>

      {uncertaintyImage ? (
        <div className="mt-5 overflow-hidden border border-white/10">
          <img src={uncertaintyImage} alt="Model uncertainty map" className="h-32 w-full object-cover opacity-90" />
        </div>
      ) : null}
    </div>
  );
}

function ImageComparisonSection({
  inputImage,
  outputImage,
  comparison,
  setComparison,
  scaleFactor,
}: {
  inputImage: string | null;
  outputImage: string | null;
  comparison: number;
  setComparison: (value: number) => void;
  scaleFactor: number | null;
}) {
  return (
    <section id="comparison" className="bg-space-950 py-16 sm:py-20">
      <div className="mx-auto max-w-[1480px] px-4 sm:px-6 lg:px-8">
        <SectionHeader eyebrow="BEFORE / AFTER EXPERIENCE" title="GEOSR IMAGE COMPARISON" copy="A draggable inspection surface for the uploaded low-resolution image and the backend-generated high-resolution preview." />
        <div className="hud-panel mt-10 overflow-hidden p-4">
          <div className="relative aspect-[16/8] min-h-[340px] overflow-hidden bg-black/40">
            {inputImage ? (
              <img src={inputImage} alt="Low-resolution input" className="absolute inset-0 h-full w-full object-cover" />
            ) : (
              <EmptyComparisonLabel text="LOW-RESOLUTION INPUT WAITING" side="left" />
            )}

            {outputImage ? (
              <div className="absolute inset-0 overflow-hidden" style={{ clipPath: `inset(0 ${100 - comparison}% 0 0)` }}>
                <img src={outputImage} alt="GeoSR AI output" className="h-full w-full object-cover" />
              </div>
            ) : (
              <EmptyComparisonLabel text="GEOSR AI OUTPUT WAITING" side="right" />
            )}

            <div className="absolute inset-y-0 z-10 w-px bg-cyan-signal shadow-[0_0_22px_rgba(0,217,255,.9)]" style={{ left: `${comparison}%` }} />
            <input
              aria-label="Compare low-resolution and GeoSR output"
              type="range"
              min="0"
              max="100"
              value={comparison}
              onChange={(event) => setComparison(Number(event.target.value))}
              className="absolute inset-x-6 bottom-6 z-20 accent-cyan-signal"
            />
            <div className="font-telemetry absolute left-5 top-5 z-20 border border-white/10 bg-black/40 px-3 py-2 text-[10px] uppercase tracking-[0.18em] text-slate-200">LOW-RESOLUTION</div>
            <div className="font-telemetry absolute right-5 top-5 z-20 border border-cyan-signal/30 bg-cyan-signal/10 px-3 py-2 text-[10px] uppercase tracking-[0.18em] text-cyan-signal">GEOSR AI OUTPUT</div>
            <div className="font-telemetry absolute left-1/2 top-1/2 z-20 -translate-x-1/2 -translate-y-1/2 border border-cyan-signal/40 bg-space-950/75 px-4 py-2 text-xs uppercase tracking-[0.18em] text-cyan-signal">
              {scaleFactor ? `${scaleFactor}x SUPER RESOLUTION` : "SUPER RESOLUTION READY"}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function SatelliteIntelligenceSection() {
  const [selected, setSelected] = useState(satelliteFleet[0]);

  return (
    <section id="satellite" className="relative overflow-hidden bg-space-900 py-16 sm:py-20">
      <div className="absolute inset-0 bg-mission-grid opacity-20" aria-hidden="true" />
      <div className="relative mx-auto max-w-[1480px] px-4 sm:px-6 lg:px-8">
        <SectionHeader eyebrow="SATELLITE INTELLIGENCE" title="ORBITAL ASSET VISUALIZATION" copy="A lightweight 3D Earth view for orbit context, mission telemetry, and satellite selection." />
        <div className="mt-10 grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_420px]">
          <div className="hud-panel relative min-h-[560px] overflow-hidden">
            <SpaceScene compact className="inset-0" />
            <div className="pointer-events-none absolute left-6 top-6 grid gap-2 font-telemetry text-[10px] uppercase tracking-[0.18em] text-cyan-signal">
              <span>LEO ORBIT</span>
              <span>MEO ORBIT</span>
              <span>GEO ORBIT</span>
            </div>
            <div className="absolute bottom-5 left-5 right-5 grid gap-2 sm:grid-cols-3">
              {satelliteFleet.map((satellite) => (
                <button
                  key={satellite.id}
                  onClick={() => setSelected(satellite)}
                  className={`border px-4 py-3 text-left font-telemetry text-xs uppercase tracking-[0.16em] transition ${
                    selected.id === satellite.id ? "border-cyan-signal bg-cyan-signal/10 text-cyan-signal" : "border-white/10 bg-black/25 text-slate-300 hover:border-cyan-signal/40"
                  }`}
                >
                  {satellite.id}
                </button>
              ))}
            </div>
          </div>

          <div className="hud-panel p-6">
            <p className="font-telemetry text-[10px] uppercase tracking-[0.24em] text-cyan-signal">SELECTED SATELLITE</p>
            <h3 className="font-display mt-3 text-3xl font-semibold uppercase tracking-[0.12em] text-white">{selected.id}</h3>
            <div className="mt-8 grid gap-4 font-telemetry text-xs uppercase">
              <TelemetryPair label="SATELLITE ID" value={selected.id} />
              <TelemetryPair label="ALTITUDE" value={selected.altitude} />
              <TelemetryPair label="VELOCITY" value={selected.velocity} />
              <TelemetryPair label="MISSION" value={selected.mission} />
              <TelemetryPair label="IMAGING STATUS" value="ACTIVE" />
              <TelemetryPair label="SIGNAL" value={selected.signal} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function MapSection({ result, scaleFactor }: { result: SuperResolutionResult | null; scaleFactor: number | null }) {
  return (
    <section id="map" className="bg-space-950 py-16 sm:py-20">
      <div className="mx-auto max-w-[1480px] px-4 sm:px-6 lg:px-8">
        <SectionHeader eyebrow="LIVE EARTH OBSERVATION MAP" title="GROUND STATION NETWORK" copy="Mission-map view showing the selected AOI, satellite footprint, ground station handoff, and GeoSR AI processing route." />
        <div className="mt-10 grid gap-5 lg:grid-cols-[minmax(0,1fr)_360px]">
          <div className="hud-panel map-grid relative min-h-[520px] overflow-hidden p-6">
            <div className="absolute left-[20%] top-[26%] h-3 w-3 rounded-full bg-cyan-signal shadow-[0_0_24px_rgba(0,217,255,.9)]" />
            <div className="absolute left-[43%] top-[46%] h-16 w-24 border border-cyan-signal bg-cyan-signal/10 shadow-glow">
              <div className="scanline absolute inset-y-0 w-1/2" />
            </div>
            <div className="absolute right-[22%] top-[30%] h-3 w-3 rounded-full bg-orbit-green shadow-[0_0_24px_rgba(50,232,117,.8)]" />
            <div className="absolute left-[20%] top-[27%] h-px w-[42%] rotate-[16deg] bg-gradient-to-r from-cyan-signal/80 to-transparent" />
            <div className="absolute right-[23%] top-[31%] h-px w-[30%] -rotate-[24deg] bg-gradient-to-r from-orbit-green/80 to-transparent" />
            <div className="absolute bottom-8 left-8 grid gap-3 font-telemetry text-[10px] uppercase tracking-[0.18em] text-slate-300">
              <span>SATELLITE - GROUND STATION - GEOSR AI - HIGH-RESOLUTION DATA</span>
              <span>ACTIVE FOOTPRINT: CENTRAL INDIA AOI</span>
            </div>
            <div className="absolute inset-8 border border-white/10" />
            <MapPin className="absolute left-[48%] top-[48%] h-7 w-7 text-cyan-signal" aria-hidden="true" />
          </div>

          <div className="hud-panel p-6">
            <p className="font-telemetry text-[10px] uppercase tracking-[0.24em] text-cyan-signal">SELECTED AOI</p>
            <h3 className="font-display mt-3 text-2xl font-semibold uppercase tracking-[0.12em] text-white">CENTRAL INDIA SCAN</h3>
            <div className="mt-8 grid gap-4 font-telemetry text-xs uppercase">
              <TelemetryPair label="LATITUDE" value="21.2514 DEG" />
              <TelemetryPair label="LONGITUDE" value="81.6296 DEG" />
              <TelemetryPair label="INPUT RESOLUTION" value={result?.input.resolution ?? "Awaiting inference metadata"} />
              <TelemetryPair label="OUTPUT RESOLUTION" value={scaleFactor ? `${10 / scaleFactor}m target view` : "Awaiting model scale"} />
              <TelemetryPair label="ENHANCEMENT" value={scaleFactor ? `${scaleFactor}x` : "Awaiting model"} />
              <TelemetryPair label="CRS" value={result?.input.crs ?? "Not reported yet"} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function AnalyticsSection({ result, modelInfo }: { result: SuperResolutionResult | null; modelInfo: ModelInfoResponse | null }) {
  const metrics = [
    { label: "PSNR", value: result?.metrics.psnr, unit: "DB" },
    { label: "SSIM", value: result?.metrics.ssim, unit: "" },
    { label: "SHARPNESS", value: null, unit: "" },
    { label: "RESOLUTION GAIN", value: result?.model.scale_factor ?? modelInfo?.scale_factor ?? null, unit: "X" },
    { label: "INFERENCE TIME", value: result?.output.inference_time_s, unit: "S" },
    { label: "MODEL PERFORMANCE", value: result?.metrics.ergas, unit: "ERGAS" },
  ];

  return (
    <section id="analytics" className="bg-space-900 py-16 sm:py-20">
      <div className="mx-auto max-w-[1480px] px-4 sm:px-6 lg:px-8">
        <SectionHeader eyebrow="EARTH OBSERVATION ANALYTICS" title="MISSION TELEMETRY GRAPHS" copy="Quality metrics are populated only when the backend computes them. Provide an HR reference image during upload to unlock PSNR, SSIM, RMSE, SAM, and ERGAS." />
        <div className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-6">
          {metrics.map((metric, index) => (
            <div key={metric.label} className="hud-panel p-4">
              <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-muted">{metric.label}</p>
              <p className="font-display mt-4 text-2xl font-semibold uppercase tracking-[0.08em] text-white">{metric.value === null || metric.value === undefined ? "N/A" : `${metric.value}${metric.unit ? ` ${metric.unit}` : ""}`}</p>
              <TelemetrySparkline active={metric.value !== null && metric.value !== undefined} offset={index} />
            </div>
          ))}
        </div>
        {!result?.metrics.available ? (
          <div className="hud-panel mt-5 p-4 text-sm leading-6 text-slate-300">
            No analytics available yet. {result?.metrics.reason ?? "Run inference with an optional HR reference image for quantitative metrics."}
          </div>
        ) : null}
      </div>
    </section>
  );
}

function TechnologySection() {
  const stages = ["SATELLITE IMAGERY", "PREPROCESSING", "DEEP LEARNING", "SUPER RESOLUTION", "GEOSPATIAL ANALYSIS"];

  return (
    <section id="technology" className="bg-space-950 py-16 sm:py-20">
      <div className="mx-auto max-w-[1480px] px-4 sm:px-6 lg:px-8">
        <SectionHeader eyebrow="GEOSR NEURAL ENGINE" title="FROM PIXELS TO PLANETARY INTELLIGENCE" copy="The interface visualizes the existing pipeline without modifying backend inference, model architecture, checkpoint loading, or geospatial output generation." />
        <div className="mt-10 grid gap-4 md:grid-cols-5">
          {stages.map((stage, index) => (
            <div key={stage} className="hud-panel min-h-44 p-5">
              <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-cyan-signal">{String(index + 1).padStart(2, "0")}</p>
              <h3 className="font-display mt-6 text-lg font-semibold uppercase tracking-[0.1em] text-white">{stage}</h3>
              <div className="mt-5 h-1 bg-white/10">
                <div className="h-full bg-cyan-signal" style={{ width: `${(index + 1) * 18}%` }} />
              </div>
            </div>
          ))}
        </div>

        <div className="mt-10 grid gap-5 lg:grid-cols-[420px_minmax(0,1fr)]">
          <div className="hud-panel p-6">
            <p className="font-telemetry text-[10px] uppercase tracking-[0.22em] text-cyan-signal">MISSION TIMELINE</p>
            <div className="mt-6 grid gap-4">
              {timeline.map((item, index) => (
                <div key={item} className="relative flex items-start gap-4">
                  <span className="mt-1 h-3 w-3 rounded-full border border-cyan-signal bg-cyan-signal/20" />
                  {index < timeline.length - 1 ? <span className="absolute left-[5px] top-6 h-7 w-px bg-cyan-signal/25" /> : null}
                  <span className="font-telemetry text-xs uppercase tracking-[0.17em] text-slate-300">{item}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="hud-panel relative min-h-[420px] overflow-hidden p-6">
            <div className="absolute inset-0 bg-mission-grid opacity-25" aria-hidden="true" />
            <div className="relative grid h-full content-center gap-8">
              <BrainNetwork large />
              <div className="mx-auto grid max-w-3xl gap-3 text-center">
                <p className="font-display text-2xl font-semibold uppercase tracking-[0.14em] text-white">INPUT IMAGE - FEATURE REPRESENTATION - SR NETWORK - IMAGE RECONSTRUCTION</p>
                <p className="text-sm leading-7 text-slate-300">The visual layer explains the ML flow while the real inference remains inside the existing GeoSR-AI backend service.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function SectionHeader({ eyebrow, title, copy }: { eyebrow: string; title: string; copy: string }) {
  return (
    <div className="max-w-3xl">
      <p className="font-telemetry text-[11px] uppercase tracking-[0.24em] text-cyan-signal">{eyebrow}</p>
      <h2 className="font-display mt-3 text-3xl font-semibold uppercase tracking-[0.12em] text-white sm:text-4xl">{title}</h2>
      <p className="mt-4 text-sm leading-7 text-slate-300 sm:text-base">{copy}</p>
    </div>
  );
}

function MiniTelemetry({ label, value, active = true }: { label: string; value: string; active?: boolean }) {
  return (
    <div className="border border-white/10 bg-white/[0.04] p-4 backdrop-blur">
      <p className="font-telemetry text-[10px] uppercase tracking-[0.2em] text-muted">{label}</p>
      <p className="font-telemetry mt-3 flex items-center gap-2 text-xs uppercase tracking-[0.15em] text-white">
        {label.includes("STATUS") ? <span className={active ? "status-dot" : "warning-dot"} /> : null}
        {value}
      </p>
    </div>
  );
}

function TelemetryPair({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-white/10 pb-2 last:border-b-0">
      <span className="text-muted">{label}</span>
      <span className="max-w-[58%] text-right text-white">{value}</span>
    </div>
  );
}

function EmptyComparisonLabel({ text, side }: { text: string; side: "left" | "right" }) {
  return (
    <div className={`absolute inset-y-0 ${side === "left" ? "left-0 w-1/2" : "right-0 w-1/2"} flex items-center justify-center bg-black/25`}>
      <span className="font-telemetry px-4 text-center text-xs uppercase tracking-[0.2em] text-slate-500">{text}</span>
    </div>
  );
}

function BrainNetwork({ large = false }: { large?: boolean }) {
  return (
    <div className={`pointer-events-none ${large ? "mx-auto h-56 max-w-3xl" : "absolute inset-7"}`} aria-hidden="true">
      <div className="relative h-full w-full">
        {Array.from({ length: large ? 18 : 9 }).map((_, index) => {
          const left = `${12 + ((index * 29) % 76)}%`;
          const top = `${16 + ((index * 37) % 68)}%`;
          return (
            <span
              key={index}
              className="absolute h-2 w-2 rounded-full bg-cyan-signal shadow-[0_0_18px_rgba(0,217,255,.85)]"
              style={{
                left,
                top,
                animation: `pulse-soft ${2.4 + (index % 5) * 0.32}s ease-in-out infinite`,
              }}
            />
          );
        })}
        {Array.from({ length: large ? 12 : 6 }).map((_, index) => (
          <span
            key={index}
            className="constellation-line absolute h-px w-[28%]"
            style={{
              left: `${10 + ((index * 17) % 72)}%`,
              top: `${18 + ((index * 23) % 66)}%`,
              transform: `rotate(${(index * 31) % 160}deg)`,
              opacity: 0.32,
            }}
          />
        ))}
      </div>
    </div>
  );
}

function TelemetrySparkline({ active, offset }: { active: boolean; offset: number }) {
  return (
    <div className="mt-5 flex h-16 items-end gap-1">
      {Array.from({ length: 18 }).map((_, index) => {
        const height = active ? 24 + ((index * 13 + offset * 9) % 40) : 10 + ((index + offset) % 5) * 3;
        return <span key={index} className={`${active ? "bg-cyan-signal/70" : "bg-white/10"} w-full`} style={{ height }} />;
      })}
    </div>
  );
}
