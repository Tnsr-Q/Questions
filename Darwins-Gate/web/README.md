# TensorQ Darwinian Web Client

One-file browser entrypoint for the full Darwinian Grid.

How to use:
1. `kubectl port-forward svc/gatewayd 443:443` (or expose publicly)
2. Open index.html
3. Watch Python code stream in live from the current Alpha cell
4. Evolved ONNX weights from Rust → PyO3 → cell → gatewayd → browser
5. WebGL renders the physics sim in real time

No build step. Pure browser rebellion.