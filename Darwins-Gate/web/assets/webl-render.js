/**
 * webl-render.js — TensorQ Darwinian WebGL2 Rendering Engine
 *
 * Handles:
 *  - WebGL2 context initialization and shader compilation
 *  - Particle system rendering (black-hole-merger, galaxy, warp-drive)
 *  - Dynamic buffer updates from Pyodide/ONNX simulation output
 *  - Chunk-type-aware code streaming integration with tensorq-client.js
 *  - Real-time stats overlay (FPS, particle count, ONNX inference latency)
 *
 * Exposed globals:
 *  - WebGLRender (class)
 *  - renderEngine (singleton instance)
 */

// ─────────────────────────────────────────────────────────────────────────────
// Shader Sources
// ─────────────────────────────────────────────────────────────────────────────

const PARTICLE_VERT_SRC = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_velocity;
layout(location = 2) in float a_mass;
layout(location = 3) in vec3 a_color;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_point_size;
uniform float u_time;

out vec3 v_color;
out float v_distance;

void main() {
    vec3 pos = a_position + a_velocity * u_time * 0.001;
    gl_Position = u_projection * u_view * vec4(pos, 1.0);
    gl_PointSize = u_point_size * (1.0 + a_mass * 0.5);
    v_color = a_color;
    v_distance = length(pos);
}`;

const PARTICLE_FRAG_SRC = `#version 300 es
precision highp float;

in vec3 v_color;
in float v_distance;

out vec4 fragColor;

void main() {
    // Circular point with soft falloff
    vec2 coord = gl_PointCoord - 0.5;
    float dist = length(coord);
    if (dist > 0.5) discard;

    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    // Distance-based glow
    float glow = exp(-v_distance * 0.3) * 0.5;
    vec3 color = v_color + glow;

    fragColor = vec4(color, alpha);
}`;

const GRID_VERT_SRC = `#version 300 es
precision highp float;

layout(location = 0) in vec3 a_position;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_time;
uniform float u_grid_scale;

void main() {
    vec3 pos = a_position * u_grid_scale;
    // Subtle wave animation
    pos.z += sin(pos.x * 2.0 + u_time * 0.5) * 0.05;
    pos.z += cos(pos.y * 2.0 + u_time * 0.3) * 0.05;
    gl_Position = u_projection * u_view * vec4(pos, 1.0);
}`;

const GRID_FRAG_SRC = `#version 300 es
precision highp float;

uniform vec3 u_grid_color;
uniform float u_grid_alpha;

out vec4 fragColor;

void main() {
    fragColor = vec4(u_grid_color, u_grid_alpha);
}`;

// ─────────────────────────────────────────────────────────────────────────────
// Matrix Utilities (no external dependency)
// ─────────────────────────────────────────────────────────────────────────────

function mat4Perspective(fovY, aspect, near, far) {
    const f = 1.0 / Math.tan(fovY / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
    ]);
}

function mat4LookAt(eye, center, up) {
    const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = 1 / Math.hypot(zx, zy, zz);
    const z = [zx * len, zy * len, zz * len];

    const xx = up[1] * z[2] - up[2] * z[1], xy = up[2] * z[0] - up[0] * z[2], xz = up[0] * z[1] - up[1] * z[0];
    len = 1 / Math.hypot(xx, xy, xz);
    const x = [xx * len, xy * len, xz * len];

    const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];

    return new Float32Array([
        x[0], y[0], z[0], 0,
        x[1], y[1], z[1], 0,
        x[2], y[2], z[2], 0,
        -(x[0] * eye[0] + x[1] * eye[1] + x[2] * eye[2]),
        -(y[0] * eye[0] + y[1] * eye[1] + y[2] * eye[2]),
        -(z[0] * eye[0] + z[1] * eye[1] + z[2] * eye[2]),
        1,
    ]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shader Compilation Helper
// ─────────────────────────────────────────────────────────────────────────────

function compileShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error("Shader compile error:", gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }
    return shader;
}

function createProgram(gl, vertSrc, fragSrc) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragSrc);
    if (!vs || !fs) return null;

    const program = gl.createProgram();
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error("Program link error:", gl.getProgramInfoLog(program));
        gl.deleteProgram(program);
        return null;
    }
    return program;
}

// ─────────────────────────────────────────────────────────────────────────────
// Particle System
// ─────────────────────────────────────────────────────────────────────────────

class ParticleSystem {
    constructor(gl, maxParticles = 100000) {
        this.gl = gl;
        this.maxParticles = maxParticles;
        this.count = 0;

        // Interleaved buffer: pos(3) + vel(3) + mass(1) + color(3) = 10 floats per particle
        this.stride = 10;
        this.buffer = new Float32Array(maxParticles * this.stride);

        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);

        this.vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferData(gl.ARRAY_BUFFER, this.buffer.byteLength, gl.DYNAMIC_DRAW);

        // Attribute layout
        let offset = 0;
        // a_position: vec3
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, this.stride * 4, offset);
        offset += 3 * 4;
        // a_velocity: vec3
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, this.stride * 4, offset);
        offset += 3 * 4;
        // a_mass: float
        gl.enableVertexAttribArray(2);
        gl.vertexAttribPointer(2, 1, gl.FLOAT, false, this.stride * 4, offset);
        offset += 1 * 4;
        // a_color: vec3
        gl.enableVertexAttribArray(3);
        gl.vertexAttribPointer(3, 3, gl.FLOAT, false, this.stride * 4, offset);

        gl.bindVertexArray(null);
    }

    /** Initialize particles with random positions in a sphere */
    init(count, radius = 1.0) {
        this.count = Math.min(count, this.maxParticles);
        for (let i = 0; i < this.count; i++) {
            const idx = i * this.stride;
            // Position: random in sphere
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            const r = radius * Math.cbrt(Math.random());
            this.buffer[idx] = r * Math.sin(phi) * Math.cos(theta);
            this.buffer[idx + 1] = r * Math.sin(phi) * Math.sin(theta);
            this.buffer[idx + 2] = r * Math.cos(phi);
            // Velocity: orbital
            const speed = 0.1 + Math.random() * 0.3;
            this.buffer[idx + 3] = -Math.sin(theta) * speed;
            this.buffer[idx + 4] = Math.cos(theta) * speed;
            this.buffer[idx + 5] = 0;
            // Mass
            this.buffer[idx + 6] = 0.5 + Math.random() * 1.5;
            // Color: teal → cyan → white gradient based on mass
            const t = this.buffer[idx + 6] / 2.0;
            this.buffer[idx + 7] = 0.0 + t * 1.0;     // R
            this.buffer[idx + 8] = 0.5 + t * 0.5;     // G
            this.buffer[idx + 9] = 0.3 + t * 0.7;     // B
        }
        this.upload();
    }

    /** Upload buffer to GPU */
    upload() {
        const gl = this.gl;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.buffer.subarray(0, this.count * this.stride));
        gl.bindBuffer(gl.ARRAY_BUFFER, null);
    }

    /** Update positions from external data (e.g., Pyodide numpy array) */
    updateFromExternal(positions, velocities) {
        const count = Math.min(positions.length / 3, velocities.length / 3, this.count);
        for (let i = 0; i < count; i++) {
            const idx = i * this.stride;
            this.buffer[idx] = positions[i * 3];
            this.buffer[idx + 1] = positions[i * 3 + 1];
            this.buffer[idx + 2] = positions[i * 3 + 2];
            this.buffer[idx + 3] = velocities[i * 3];
            this.buffer[idx + 4] = velocities[i * 3 + 1];
            this.buffer[idx + 5] = velocities[i * 3 + 2];
        }
        this.upload();
    }

    /** Simple CPU physics step (orbital rotation + slight decay) */
    step(dt) {
        for (let i = 0; i < this.count; i++) {
            const idx = i * this.stride;
            const px = this.buffer[idx], py = this.buffer[idx + 1];
            const vx = this.buffer[idx + 3], vy = this.buffer[idx + 4];
            const dist = Math.hypot(px, py) + 0.001;
            // Gravitational acceleration toward center
            const accel = -0.5 / (dist * dist);
            const ax = (-px / dist) * accel;
            const ay = (-py / dist) * accel;
            this.buffer[idx + 3] += ax * dt;
            this.buffer[idx + 4] += ay * dt;
            this.buffer[idx] += this.buffer[idx + 3] * dt;
            this.buffer[idx + 1] += this.buffer[idx + 4] * dt;
        }
        this.upload();
    }

    draw(program, gl) {
        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.POINTS, 0, this.count);
        gl.bindVertexArray(null);
    }

    destroy(gl) {
        gl.deleteVertexArray(this.vao);
        gl.deleteBuffer(this.vbo);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid Renderer (spatial reference)
// ─────────────────────────────────────────────────────────────────────────────

class GridRenderer {
    constructor(gl, gridSize = 20) {
        this.gl = gl;
        const vertices = [];
        const half = gridSize / 2;
        for (let i = -half; i <= half; i++) {
            // X lines
            vertices.push(-half, 0, i, half, 0, i);
            // Z lines
            vertices.push(i, 0, -half, i, 0, half);
        }
        this.vertexCount = vertices.length / 3;
        const vbo = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
        this.vao = vao;
        this.vbo = vbo;
    }

    draw(program, gl) {
        gl.bindVertexArray(this.vao);
        gl.drawArrays(gl.LINES, 0, this.vertexCount);
        gl.bindVertexArray(null);
    }

    destroy(gl) {
        gl.deleteVertexArray(this.vao);
        gl.deleteBuffer(this.vbo);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stats Overlay
// ─────────────────────────────────────────────────────────────────────────────

class RenderStats {
    constructor() {
        this.fps = 0;
        this.frameCount = 0;
        this.lastFpsTime = performance.now();
        this.particleCount = 0;
        this.onnxLatencyMs = 0;
        this.chunksReceived = 0;
        this.chunksExecuted = 0;
        this.chunkType = "NONE";
        this.simStatus = "INITIALIZING";
    }

    tick() {
        this.frameCount++;
        const now = performance.now();
        if (now - this.lastFpsTime >= 1000) {
            this.fps = this.frameCount;
            this.frameCount = 0;
            this.lastFpsTime = now;
        }
    }

    updateDOM(el) {
        if (!el) return;
        el.textContent =
            `[TensorQ Darwinian Grid]\n` +
            `  FPS:          ${this.fps}\n` +
            `  Particles:    ${this.particleCount}\n` +
            `  ONNX Latency: ${this.onnxLatencyMs.toFixed(1)} ms\n` +
            `  Chunks:       ${this.chunksReceived} recv / ${this.chunksExecuted} exec\n` +
            `  Last Chunk:   ${this.chunkType}\n` +
            `  Sim Status:   ${this.simStatus}`;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Render Engine
// ─────────────────────────────────────────────────────────────────────────────

class WebGLRender {
    constructor(canvas, opts = {}) {
        this.canvas = canvas;
        this.gl = canvas.getContext("webgl2", {
            antialias: opts.antialias !== false,
            alpha: opts.alpha !== false,
            premultipliedAlpha: false,
            preserveDrawingBuffer: false,
        }) || canvas.getContext("webgl", {
            antialias: opts.antialias !== false,
            alpha: opts.alpha !== false,
        });

        if (!this.gl) {
            throw new Error("WebGL not supported");
        }

        this.gl2 = this.gl instanceof WebGL2RenderingContext;
        this.stats = new RenderStats();
        this.statusEl = document.getElementById("status");
        this.running = false;
        this.animFrameId = null;
        this.startTime = performance.now();

        // Camera
        this.cameraDistance = opts.cameraDistance || 5.0;
        this.cameraTheta = opts.cameraTheta || 0.3;
        this.cameraPhi = opts.cameraPhi || 0.8;
        this.cameraTarget = opts.cameraTarget || [0, 0, 0];

        // Simulation config
        this.maxParticles = opts.maxParticles || 10000;
        this.gridSize = opts.gridSize || 20;
        this.enableGrid = opts.enableGrid !== false;
        this.enablePhysics = opts.enablePhysics !== false;
        this.physicsStepMs = opts.physicsStepMs || 16;

        // Chunk streaming state
        this.chunkBuffer = "";
        this.pendingExecution = false;
        this.onChunkCallback = null;

        // ONNX session reference (set externally by tensorq-client.js)
        this.onnxSession = null;

        this._init();
    }

    _init() {
        const gl = this.gl;

        // Enable blending for particles
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(false);

        // Compile programs
        this.particleProgram = createProgram(gl, PARTICLE_VERT_SRC, PARTICLE_FRAG_SRC);
        if (!this.particleProgram) throw new Error("Failed to compile particle shaders");
        this.particleUniforms = this._cacheUniforms(this.particleProgram, [
            "u_projection", "u_view", "u_point_size", "u_time",
        ]);

        if (this.enableGrid) {
            this.gridProgram = createProgram(gl, GRID_VERT_SRC, GRID_FRAG_SRC);
            if (!this.gridProgram) throw new Error("Failed to compile grid shaders");
            this.gridUniforms = this._cacheUniforms(this.gridProgram, [
                "u_projection", "u_view", "u_time", "u_grid_scale", "u_grid_color", "u_grid_alpha",
            ]);
        }

        // Initialize particle system
        this.particles = new ParticleSystem(gl, this.maxParticles);
        this.particles.init(this.maxParticles, 2.0);
        this.stats.particleCount = this.maxParticles;

        // Initialize grid
        if (this.enableGrid) {
            this.grid = new GridRenderer(gl, this.gridSize);
        }

        // Handle resize
        this._resizeHandler = () => this._onResize();
        window.addEventListener("resize", this._resizeHandler);
        this._onResize();
    }

    _cacheUniforms(program, names) {
        const gl = this.gl;
        const cache = {};
        for (const name of names) {
            cache[name] = gl.getUniformLocation(program, name);
        }
        return cache;
    }

    _onResize() {
        const dpr = window.devicePixelRatio || 1;
        const w = this.canvas.clientWidth * dpr;
        const h = this.canvas.clientHeight * dpr;
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
            this.gl.viewport(0, 0, w, h);
        }
        this.aspect = w / h;
        this.projection = mat4Perspective(Math.PI / 4, this.aspect, 0.1, 100.0);
    }

    /** Set the ONNX session from externally loaded model */
    setONNXSession(session) {
        this.onnxSession = session;
        this.stats.simStatus = "ONNX_LOADED";
    }

    /** Update status text */
    setStatus(msg) {
        if (this.statusEl) {
            this.statusEl.textContent = msg;
        }
    }

    // ── Chunk Streaming Integration ─────────────────────────────────────────

    /**
     * Called by tensorq-client.js when a CodeChunk arrives.
     * @param {Object} chunk - CodeChunk proto fields:
     *   { content: string, type: ChunkType, sequenceId: number }
     */
    onChunk(chunk) {
        this.stats.chunksReceived++;
        this.chunkBuffer += chunk.content;

        const typeMap = {
            0: "UNSPECIFIED",
            1: "PREAMBLE",
            2: "DEFINITION",
            3: "EXECUTION",
        };
        this.stats.chunkType = typeMap[chunk.type] || "UNKNOWN";

        // Notify external callback (tensorq-client.js)
        if (this.onChunkCallback) {
            this.onChunkCallback(chunk, this.chunkBuffer);
        }

        // Auto-trigger execution on EXECUTION chunk
        if (chunk.type === 3) { // CHUNK_TYPE_EXECUTION
            this.pendingExecution = true;
        }
    }

    /** Execute the accumulated chunk buffer via Pyodide (called from render loop) */
    async executePendingChunks(pyodide) {
        if (!this.pendingExecution || !pyodide) return;
        this.pendingExecution = false;
        try {
            await pyodide.runPythonAsync(this.chunkBuffer);
            this.stats.chunksExecuted++;
            this.stats.simStatus = "RUNNING";
        } catch (e) {
            console.warn("Chunk execution failed:", e);
            this.stats.simStatus = "EXEC_ERROR";
        }
    }

    // ── Camera Controls ────────────────────────────────────────────────────

    /** Set camera spherical coordinates */
    setCamera(theta, phi, distance) {
        this.cameraTheta = theta;
        this.cameraPhi = phi;
        this.cameraDistance = distance;
    }

    /** Orbit camera by delta (for mouse/touch) */
    orbitCamera(dTheta, dPhi) {
        this.cameraTheta += dTheta;
        this.cameraPhi = Math.max(0.1, Math.min(Math.PI - 0.1, this.cameraPhi + dPhi));
    }

    /** Zoom camera */
    zoomCamera(delta) {
        this.cameraDistance = Math.max(1.0, Math.min(50.0, this.cameraDistance + delta));
    }

    _getViewMatrix() {
        const r = this.cameraDistance;
        const theta = this.cameraTheta;
        const phi = this.cameraPhi;
        const eye = [
            this.cameraTarget[0] + r * Math.sin(phi) * Math.cos(theta),
            this.cameraTarget[1] + r * Math.cos(phi),
            this.cameraTarget[2] + r * Math.sin(phi) * Math.sin(theta),
        ];
        return mat4LookAt(eye, this.cameraTarget, [0, 1, 0]);
    }

    // ── Render Loop ────────────────────────────────────────────────────────

    /** Start the render loop */
    start() {
        if (this.running) return;
        this.running = true;
        this._render();
    }

    /** Stop the render loop */
    stop() {
        this.running = false;
        if (this.animFrameId) {
            cancelAnimationFrame(this.animFrameId);
            this.animFrameId = null;
        }
    }

    _render = () => {
        if (!this.running) return;
        this.animFrameId = requestAnimationFrame(this._render);

        const gl = this.gl;
        const time = performance.now() - this.startTime;
        this.stats.tick();

        // Physics step
        if (this.enablePhysics) {
            this.particles.step(this.physicsStepMs * 0.001);
        }

        // Clear
        gl.clearColor(0.0, 0.0, 0.02, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const view = this._getViewMatrix();

        // Draw grid
        if (this.enableGrid && this.gridProgram) {
            gl.useProgram(this.gridProgram);
            gl.uniformMatrix4fv(this.gridUniforms.u_projection, false, this.projection);
            gl.uniformMatrix4fv(this.gridUniforms.u_view, false, view);
            gl.uniform1f(this.gridUniforms.u_time, time * 0.001);
            gl.uniform1f(this.gridUniforms.u_grid_scale, 1.0);
            gl.uniform3f(this.gridUniforms.u_grid_color, 0.05, 0.15, 0.1);
            gl.uniform1f(this.gridUniforms.u_grid_alpha, 0.4);
            gl.depthMask(false);
            this.grid.draw(this.gridProgram, gl);
        }

        // Draw particles
        gl.useProgram(this.particleProgram);
        gl.uniformMatrix4fv(this.particleUniforms.u_projection, false, this.projection);
        gl.uniformMatrix4fv(this.particleUniforms.u_view, false, view);
        gl.uniform1f(this.particleUniforms.u_point_size, 4.0);
        gl.uniform1f(this.particleUniforms.u_time, time);
        gl.depthMask(false);
        this.particles.draw(this.particleProgram, gl);

        // Update stats overlay
        this.stats.updateDOM(this.statusEl);
    };

    // ── External Data Integration ───────────────────────────────────────────

    /**
     * Called by Pyodide (main.py) to update particle positions/velocities.
     * Exposed as global: renderEngine.updateParticles(positions, velocities)
     * where positions and velocities are Float32Array from js module.
     */
    updateParticles(positions, velocities) {
        this.particles.updateFromExternal(positions, velocities);
        this.stats.particleCount = positions.length / 3;
    }

    /** Run a single ONNX inference step (called from Python or render loop) */
    async runInference(inputData) {
        if (!this.onnxSession) return null;
        const start = performance.now();
        const results = await this.onnxSession.run(null, inputData);
        this.stats.onnxLatencyMs = performance.now() - start;
        return results;
    }

    // ── Cleanup ─────────────────────────────────────────────────────────────

    destroy() {
        this.stop();
        window.removeEventListener("resize", this._resizeHandler);
        if (this.particles) this.particles.destroy(this.gl);
        if (this.grid) this.grid.destroy(this.gl);
        if (this.particleProgram) this.gl.deleteProgram(this.particleProgram);
        if (this.gridProgram) this.gl.deleteProgram(this.gridProgram);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Global Singleton — accessible from Pyodide and tensorq-client.js
// ─────────────────────────────────────────────────────────────────────────────

let renderEngine = null;

/**
 * Initialize the render engine (call once on window.onload).
 * @param {HTMLCanvasElement} canvas
 * @param {Object} opts — same as WebGLRender constructor options
 * @returns {WebGLRender}
 */
function initRenderEngine(canvas, opts) {
    renderEngine = new WebGLRender(canvas, opts);
    renderEngine.start();
    return renderEngine;
}

// Auto-init if canvas exists and no script has initialized yet
if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", () => {
        const canvas = document.getElementById("canvas");
        if (canvas && !renderEngine) {
            initRenderEngine(canvas);
        }
    });
}

// Export for ES module usage (when imported via type="module")
// renderEngine is exported as a live binding — it will be non-null after
// initRenderEngine() runs (or after auto-init on DOMContentLoaded).
export { WebGLRender, ParticleSystem, GridRenderer, RenderStats, initRenderEngine, renderEngine };
export default WebGLRender;
