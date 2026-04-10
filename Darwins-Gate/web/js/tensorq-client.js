/**
 * tensorq-client.js — Darwinian Gateway ConnectRPC Client
 *
 * Connects to the CortexGateway gRPC-Web service, streams CodeChunk
 * payloads into the Pyodide runtime, and coordinates with the WebGL2
 * render engine (webl-render.js).
 *
 * Data flow:
 *   gatewayd (ConnectRPC :8080)
 *     → StreamSimulation(SimRequest)
 *       → CodeChunk stream
 *         → pyodide.runPythonAsync()  [main.py environment]
 *           → physics_step() → update_webgl() → renderEngine
 *             → GPU render loop @ 60fps
 */

import { createConnectTransport } from "@connectrpc/connect-web";
import { createPromiseClient } from "@connectrpc/connect";
// Import the init function and the module reference — renderEngine is a
// live binding that gets set when initRenderEngine() runs (ESM live export).
import { initRenderEngine } from "../assets/webl-render.js";
import * as weblRender from "../assets/webl-render.js";

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

const GATEWAY_URL = import.meta.env?.VITE_GATEWAY_URL ?? "https://localhost:8080";
const DEFAULT_MODEL = "black-hole-merger";
const DEFAULT_HYPER_PARAMS = { gravity: "1.0", drag: "0.01", softening: "0.05" };

// ─────────────────────────────────────────────────────────────────────────────
// Transport & Client
// ─────────────────────────────────────────────────────────────────────────────

const transport = createConnectTransport({
    baseUrl: GATEWAY_URL,
    useBinaryFormat: true,
    // Allow self-signed certs in development
    fetchInit: {
        // In production, use proper TLS
    },
});

// Lazy client — created once Pyodide is ready
let client = null;

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

let pyodide = null;
const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
let render = null;
let isStreaming = false;
let abortController = null;

// ─────────────────────────────────────────────────────────────────────────────
// Pyodide Initialization
// ─────────────────────────────────────────────────────────────────────────────

async function initPyodide() {
    setStatus("Loading Pyodide + NumPy…");

    pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.29.1/full/",
    });

    await pyodide.loadPackage(["numpy", "micropip"]);

    // Install onnxruntime-web if not pre-loaded
    try {
        await pyodide.loadPackagesFromImports(await (await fetch("pyodide/main.py")).text());
    } catch (_) {
        // Expected if no Python imports need micropip packages
    }

    // Expose renderEngine singleton to Pyodide before main.py runs
    // The webl-render.js module exports `renderEngine` which is set after init
    pyodide.globals.set("renderEngine", null); // placeholder, set after init

    // Load and execute the main.py environment
    const mainPy = await (await fetch("pyodide/main.py")).text();
    await pyodide.runPythonAsync(mainPy);

    setStatus("Pyodide + Darwinian environment ready");
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Start streaming simulation from the current Alpha cell.
 *
 * @param {string} modelId - Simulation model identifier
 * @param {Object} hyperParams - Hyperparameter key-value pairs
 */
async function startStreaming(modelId = DEFAULT_MODEL, hyperParams = DEFAULT_HYPER_PARAMS) {
    if (isStreaming) {
        console.warn("[TensorQ] Already streaming — stopping first");
        stopStreaming();
    }

    if (!client) {
        console.error("[TensorQ] ConnectRPC client not initialized");
        setStatus("ERROR: Gateway client not ready");
        return;
    }

    isStreaming = true;
    abortController = new AbortController();
    setStatus(`Streaming Alpha cell: ${modelId}…`);

    // Notify Pyodide that a simulation is starting
    if (pyodide) {
        try {
            const hpDict = pyodide.globals.get("dict")();
            for (const [k, v] of Object.entries(hyperParams)) {
                hpDict.set(k, v);
            }
            const onStart = pyodide.globals.get("on_simulation_start");
            if (onStart) {
                // Convert hyperParams to a Python dict
                const pyHp = {};
                for (const [k, v] of Object.entries(hyperParams)) {
                    pyHp[k] = v;
                }
                pyodide.runPythonAsync(`on_simulation_start("${modelId}", ${JSON.stringify(pyHp).replace(/"/g, "'")})`);
            }
        } catch (_) {
            // Non-critical — simulation will use defaults
        }
    }

    try {
        const stream = client.streamSimulation({
            modelId,
            hyperParameters: hyperParams,
            chunkSizeHint: 8192,
        }, { signal: abortController.signal });

        let buffer = "";
        let totalChunks = 0;

        for await (const chunk of stream) {
            if (!isStreaming) break;
            totalChunks++;

            // Forward to render engine for stats tracking
            if (weblRender.renderEngine) {
                weblRender.renderEngine.onChunk({
                    content: chunk.content || "",
                    type: chunk.type || 0,
                    sequenceId: Number(chunk.sequenceId || 0),
                });
            }

            buffer += chunk.content || "";

            // Execute on EXECUTION chunk (type=3) or double-newline boundary
            if (chunk.type === 3 || (chunk.content || "").includes("\n\n")) {
                try {
                    await pyodide.runPythonAsync(buffer);
                    if (weblRender.renderEngine) {
                        weblRender.renderEngine.stats.chunksExecuted++;
                        weblRender.renderEngine.stats.simStatus = "RUNNING";
                    }
                    setStatus(`Executed chunk #${chunk.sequenceId} — ${totalChunks} total — live sim updating`);
                } catch (e) {
                    console.warn("[TensorQ] Partial execution failed:", e);
                    if (weblRender.renderEngine) {
                        weblRender.renderEngine.stats.simStatus = "EXEC_ERROR";
                    }
                }
            }

            // ONNX weights can arrive as base64-encoded content in a dedicated chunk.
            // Convention: if content starts with "ONNX_BASE64:", decode and load.
            const content = chunk.content || "";
            if (content.startsWith("ONNX_BASE64:")) {
                await loadOnnxFromBase64(content.slice(12));
            }
        }

        setStatus(`Stream complete: ${totalChunks} chunks from ${modelId}`);
    } catch (e) {
        if (e.name === "AbortError") {
            setStatus("Streaming aborted");
        } else {
            console.error("[TensorQ] Stream error:", e);
            setStatus(`Stream error: ${e.message}`);
        }
    } finally {
        isStreaming = false;
    }
}

/**
 * Stop the active streaming connection.
 */
function stopStreaming() {
    isStreaming = false;
    if (abortController) {
        abortController.abort();
        abortController = null;
    }
    setStatus("Streaming stopped");
}

/**
 * Load ONNX weights from a base64-encoded string in the stream.
 */
async function loadOnnxFromBase64(base64Data) {
    try {
        const ort = window.ort;
        if (!ort) {
            console.warn("[TensorQ] ONNX Runtime Web not loaded");
            return;
        }

        // Decode base64 to Uint8Array
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        const session = await ort.InferenceSession.create(bytes, {
            executionProviders: ["webgpu"],
        });

        // Expose to Pyodide Python runtime
        pyodide.globals.set("onnx_session", session);

        // Push to render engine
        if (weblRender.renderEngine) {
            weblRender.renderEngine.setONNXSession(session);
        }

        setStatus(`ONNX model loaded — ${bytes.byteLength} bytes`);
    } catch (e) {
        console.error("[TensorQ] ONNX load failed:", e);
    }
}

// Keep legacy alias for backward compatibility
async function loadOnnxFromStream(weightsBytes) {
    // Convert raw bytes to base64 and reuse the base64 path
    const binary = String.fromCharCode(...weightsBytes);
    const b64 = btoa(binary);
    await loadOnnxFromBase64(b64);
}

// ─────────────────────────────────────────────────────────────────────────────
// Camera Controls
// ─────────────────────────────────────────────────────────────────────────────

function setupCameraControls() {
    if (!canvas) return;

    let isDragging = false;
    let lastX = 0, lastY = 0;

    canvas.addEventListener("mousedown", (e) => {
        isDragging = true;
        lastX = e.clientX;
        lastY = e.clientY;
    });

    canvas.addEventListener("mousemove", (e) => {
        if (!isDragging || !weblRender.renderEngine) return;
        const dx = (e.clientX - lastX) * 0.005;
        const dy = (e.clientY - lastY) * 0.005;
        weblRender.renderEngine.orbitCamera(dx, dy);
        lastX = e.clientX;
        lastY = e.clientY;
    });

    canvas.addEventListener("mouseup", () => { isDragging = false; });
    canvas.addEventListener("mouseleave", () => { isDragging = false; });

    canvas.addEventListener("wheel", (e) => {
        if (!weblRender.renderEngine) return;
        weblRender.renderEngine.zoomCamera(e.deltaY * 0.005);
        e.preventDefault();
    }, { passive: false });

    // Touch support
    let lastTouchDist = 0;
    canvas.addEventListener("touchstart", (e) => {
        if (e.touches.length === 1) {
            isDragging = true;
            lastX = e.touches[0].clientX;
            lastY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
            lastTouchDist = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY,
            );
        }
        e.preventDefault();
    }, { passive: false });

    canvas.addEventListener("touchmove", (e) => {
        if (!weblRender.renderEngine) return;
        if (e.touches.length === 1 && isDragging) {
            const dx = (e.touches[0].clientX - lastX) * 0.005;
            const dy = (e.touches[0].clientY - lastY) * 0.005;
            weblRender.renderEngine.orbitCamera(dx, dy);
            lastX = e.touches[0].clientX;
            lastY = e.touches[0].clientY;
        } else if (e.touches.length === 2) {
            const dist = Math.hypot(
                e.touches[0].clientX - e.touches[1].clientX,
                e.touches[0].clientY - e.touches[1].clientY,
            );
            weblRender.renderEngine.zoomCamera((lastTouchDist - dist) * 0.02);
            lastTouchDist = dist;
        }
        e.preventDefault();
    }, { passive: false });

    canvas.addEventListener("touchend", () => { isDragging = false; });
}

// ─────────────────────────────────────────────────────────────────────────────
// Status
// ─────────────────────────────────────────────────────────────────────────────

function setStatus(msg) {
    if (statusEl) {
        statusEl.textContent = msg;
    }
    // Also update loading overlay if still visible
    const loadingText = document.getElementById("loading-text");
    if (loadingText) {
        loadingText.textContent = msg;
    }
    console.log(`[TensorQ] ${msg}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API (exposed on window for external control)
// ─────────────────────────────────────────────────────────────────────────────

window.TensorQClient = {
    startStreaming,
    stopStreaming,
    initPyodide,
    get isReady() {
        return pyodide !== null && renderEngine !== null;
    },
    get isStreamingActive() {
        return isStreaming;
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// Window Onload — Full Initialization
// ─────────────────────────────────────────────────────────────────────────────

window.onload = async () => {
    // 1. Initialize WebGL2 render engine
    if (canvas) {
        render = initRenderEngine(canvas, {
            maxParticles: 10000,
            enableGrid: true,
            enablePhysics: true,
            cameraDistance: 5.0,
        });

        // Now set the renderEngine reference in Pyodide
        setupCameraControls();
    }

    // 2. Initialize Pyodide + main.py environment
    await initPyodide();

    // 3. Set the actual renderEngine reference in Pyodide (after init)
    if (pyodide && weblRender.renderEngine) {
        pyodide.globals.set("renderEngine", weblRender.renderEngine);
    }

    // 4. Initialize ConnectRPC client (after Pyodide so proto types are ready)
    try {
        const { DarwinianGateway } = await import("../gen/go/darwinianv1/darwinian_gateway_pb.js");
        client = createPromiseClient(DarwinianGateway, transport);
        setStatus("Darwinian Gateway connected — starting stream…");

        // 5. Start streaming from the Alpha cell
        await startStreaming();
    } catch (e) {
        console.warn("[TensorQ] Gateway not available — running in standalone mode:", e);
        setStatus("Gateway unavailable — running local simulation only");

        // Run simulation without gateway — pure local physics + rendering
        if (pyodide) {
            pyodide.runPythonAsync(`
init_particles(10000)
print("[TensorQ] Running standalone simulation")
`);
        }
    }
};
