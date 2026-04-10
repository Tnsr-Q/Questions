import time
import random
import json
import os
import uuid
import Pyro5.api
from kubernetes import client, config

# Load "God Mode" credentials (in-cluster)
try:
    config.load_incluster_config()
except:
    config.load_kube_config()  # For local testing

v1 = client.CoreV1Api()
NAMESPACE = os.getenv("NAMESPACE", "default")
PYRO_NS_URI = "PYRO:Pyro.NameServer@pyro-ns:9090"

def spawn_pod(genome, generation):
    """
    Directly calls the K8s API to spawn a new 'Cell' pod.
    """
    unique_id = str(uuid.uuid4())[:8]
    pod_name = f"solver-{generation}-{unique_id}"
    
    # The Genome is passed as a JSON string in the ENV vars
    genome_json = json.dumps(genome)

    pod_manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "labels": {
                "app": "solver-cell",
                "generation": str(generation),
                "species": "gravity-v1"
            }
        },
        "spec": {
            "containers": [{
                "name": "solver",
                "image": "my-repo/solver-cell:latest", # The Image from Phase 3
                "env": [
                    {"name": "GENOME", "value": genome_json},
                    {"name": "PYRO_NS_HOST", "value": "pyro-ns"},
                    {"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}} 
                ]
            }],
            "restartPolicy": "Never" # If it dies, we spawn a new one manually
        }
    }

    print(f"🧬 Spawning Mutant: {pod_name}")
    v1.create_namespaced_pod(body=pod_manifest, namespace=NAMESPACE)

def kill_pod(pod_name):
    print(f"💀 Reaping weakling: {pod_name}")
    v1.delete_namespaced_pod(name=pod_name, namespace=NAMESPACE)

def get_fitness_scores():
    """
    Connects to the Pyro Name Server to find all active agents,
    then queries them for their latest fitness score.
    """
    scores = []
    ns = Pyro5.api.Proxy(PYRO_NS_URI)
    
    # Get all registered objects starting with 'solver.'
    solvers = ns.list(prefix="solver.")
    
    for name, uri in solvers.items():
        try:
            # Connect to the cell and ask "How are you doing?"
            with Pyro5.api.Proxy(uri) as p:
                scores.append({
                    "name": name, 
                    "fitness": p.fitness,  # The Agent tracks its own score
                    "genome": p.get_genome(),
                    "pod_name": p.pod_name
                })
        except:
            # Node might be dead/unreachable
            pass
    return scores

def evolution_cycle():
    while True:
        print("👁️ Overseer scanning swarm...")
        population = get_fitness_scores()
        
        if len(population) < 5:
             # Genesis: If empty, spawn randoms
            print("🌱 Seeding initial population...")
            spawn_pod(genome={"gravity": 9.8, "drag": 0.5}, generation=0)
        
        elif population:
            # 1. Sort by fitness
            population.sort(key=lambda x: x['fitness'], reverse=True)
            
            # 2. Kill the bottom 50%
            cull_count = int(len(population) * 0.5)
            victims = population[-cull_count:]
            survivors = population[:-cull_count]
            
            for v in victims:
                kill_pod(v['pod_name'])

            # 3. Reproduce from the top 10% (Elitism)
            top_performers = survivors[:2] 
            
            # Fill the void
            for i in range(cull_count):
                parent = random.choice(top_performers)
                
                # MUTATION LOGIC
                new_genome = parent['genome'].copy()
                new_genome['gravity'] += random.uniform(-0.5, 0.5) # Mutate
                
                spawn_pod(new_genome, generation=parent.get('gen', 0) + 1)
        
        time.sleep(10) # Evolution Tick

if __name__ == "__main__":
    evolution_cycle()
