from tensorq_rust import evolve_model

def solve_physics(self, inputs):
    onnx_bytes = evolve_model(list(self.genome.values()), inputs)
    # Return bytes → streamed back through gatewayd → Pyodide
    return {"model_weights": onnx_bytes}