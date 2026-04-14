/**
 * SystemHealthDashboard - Real-time metrics widget for Darwin's Gate
 * Polls /metrics endpoint and displays eBPF + Swarm health status
 */
class SystemHealthDashboard {
    constructor(options = {}) {
        this.metricsUrl = options.metricsUrl || '/metrics';
        this.pollInterval = options.pollInterval || 2000; // 2 seconds
        this.containerId = options.containerId || 'system-health-dashboard';
        this.poller = null;
        this.lastMetrics = {};
        
        this.init();
    }

    init() {
        this.createWidget();
        this.startPolling();
    }

    createWidget() {
        // Create floating dashboard container
        const dashboard = document.createElement('div');
        dashboard.id = this.containerId;
        dashboard.innerHTML = `
            <div class="dashboard-header">
                <span class="dashboard-title">🔬 SYSTEM HEALTH</span>
                <span class="dashboard-status" id="dashboard-status">CONNECTING</span>
            </div>
            
            <div class="dashboard-section">
                <div class="section-title">⚡ eBPF DATAPATH</div>
                <div class="metric-row">
                    <span class="metric-label">Map Entries:</span>
                    <span class="metric-value" id="ebpf-map-entries">-</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Hit Rate:</span>
                    <span class="metric-value" id="ebpf-hit-rate">-</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Route Hits:</span>
                    <span class="metric-value" id="ebpf-hits">-</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Route Misses:</span>
                    <span class="metric-value" id="ebpf-misses">-</span>
                </div>
            </div>
            
            <div class="dashboard-section">
                <div class="section-title">🌐 SWARM BRIDGE</div>
                <div class="metric-row">
                    <span class="metric-label">Leader Health:</span>
                    <span class="metric-value" id="swarm-health">-</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Fetch Success:</span>
                    <span class="metric-value" id="swarm-success">-</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Fetch Failures:</span>
                    <span class="metric-value" id="swarm-failures">-</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Timeouts:</span>
                    <span class="metric-value" id="swarm-timeouts">-</span>
                </div>
            </div>
            
            <div class="dashboard-footer">
                <small>Last update: <span id="last-update">-</span></small>
            </div>
        `;
        
        document.body.appendChild(dashboard);
        this.applyStyles();
    }

    applyStyles() {
        const styles = `
            #${this.containerId} {
                position: fixed;
                top: 20px;
                right: 20px;
                width: 320px;
                background: rgba(10, 15, 30, 0.95);
                border: 1px solid #00ff88;
                border-radius: 8px;
                padding: 15px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                color: #00ff88;
                z-index: 10000;
                box-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
                backdrop-filter: blur(10px);
            }
            
            .dashboard-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #00ff88;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            
            .dashboard-title {
                font-weight: bold;
                font-size: 14px;
                letter-spacing: 1px;
            }
            
            .dashboard-status {
                padding: 4px 8px;
                border-radius: 4px;
                background: #333;
                font-size: 10px;
            }
            
            .dashboard-status.connected {
                background: #00ff88;
                color: #000;
            }
            
            .dashboard-status.degraded {
                background: #ffaa00;
                color: #000;
            }
            
            .dashboard-status.error {
                background: #ff4444;
                color: #fff;
            }
            
            .dashboard-section {
                margin-bottom: 15px;
            }
            
            .section-title {
                font-weight: bold;
                margin-bottom: 10px;
                color: #00ffff;
                font-size: 13px;
            }
            
            .metric-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 6px;
                padding: 4px 0;
            }
            
            .metric-label {
                color: #888;
            }
            
            .metric-value {
                font-weight: bold;
                color: #00ff88;
            }
            
            .dashboard-footer {
                border-top: 1px solid #333;
                padding-top: 10px;
                margin-top: 10px;
                color: #666;
                text-align: right;
            }
        `;
        
        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    startPolling() {
        this.poller = setInterval(() => this.fetchMetrics(), this.pollInterval);
        this.fetchMetrics(); // Initial fetch
    }

    stopPolling() {
        if (this.poller) {
            clearInterval(this.poller);
            this.poller = null;
        }
    }

    async fetchMetrics() {
        try {
            const response = await fetch(this.metricsUrl);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            
            const text = await response.text();
            const metrics = this.parsePrometheusMetrics(text);
            this.updateDisplay(metrics);
            this.setStatus('connected');
            this.lastMetrics = metrics;
        } catch (error) {
            console.warn('Metrics fetch failed:', error.message);
            this.setStatus('error');
        }
    }

    parsePrometheusMetrics(text) {
        const metrics = {};
        const lines = text.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('#') || !line.trim()) continue;
            
            const match = line.match(/^([a-zA-Z_:]+)\s+([0-9.eE+-]+)$/);
            if (match) {
                const [, name, value] = match;
                metrics[name] = parseFloat(value);
            }
        }
        
        return metrics;
    }

    updateDisplay(metrics) {
        // eBPF metrics
        const mapEntries = metrics['tensorq_ebpf_map_entries'] || 0;
        const hits = metrics['tensorq_ebpf_route_hits_total'] || 0;
        const misses = metrics['tensorq_ebpf_route_misses_total'] || 0;
        const hitRate = hits + misses > 0 ? ((hits / (hits + misses)) * 100).toFixed(1) : 0;
        
        document.getElementById('ebpf-map-entries').textContent = mapEntries;
        document.getElementById('ebpf-hits').textContent = hits.toLocaleString();
        document.getElementById('ebpf-misses').textContent = misses.toLocaleString();
        document.getElementById('ebpf-hit-rate').textContent = `${hitRate}%`;
        
        // Color-code hit rate
        const hitRateEl = document.getElementById('ebpf-hit-rate');
        if (parseFloat(hitRate) >= 95) {
            hitRateEl.style.color = '#00ff88';
        } else if (parseFloat(hitRate) >= 80) {
            hitRateEl.style.color = '#ffaa00';
        } else {
            hitRateEl.style.color = '#ff4444';
        }
        
        // Swarm metrics
        const success = metrics['tensorq_swarm_leader_fetch_success_total'] || 0;
        const failures = metrics['tensorq_swarm_leader_fetch_failure_total'] || 0;
        const timeouts = metrics['tensorq_swarm_proposal_fetch_timeout_total'] || 0;
        const healthRatio = metrics['tensorq_swarm_leader_health_ratio'] || 0;
        
        document.getElementById('swarm-success').textContent = success.toLocaleString();
        document.getElementById('swarm-failures').textContent = failures.toLocaleString();
        document.getElementById('swarm-timeouts').textContent = timeouts.toLocaleString();
        document.getElementById('swarm-health').textContent = `${(healthRatio * 100).toFixed(1)}%`;
        
        // Color-code health ratio
        const healthEl = document.getElementById('swarm-health');
        if (healthRatio >= 0.95) {
            healthEl.style.color = '#00ff88';
        } else if (healthRatio >= 0.80) {
            healthEl.style.color = '#ffaa00';
        } else {
            healthEl.style.color = '#ff4444';
        }
        
        // Update timestamp
        const now = new Date();
        document.getElementById('last-update').textContent = now.toLocaleTimeString();
    }

    setStatus(status) {
        const statusEl = document.getElementById('dashboard-status');
        statusEl.className = 'dashboard-status ' + status;
        statusEl.textContent = status.toUpperCase();
    }

    getLastMetrics() {
        return this.lastMetrics;
    }
}

// Auto-initialize when DOM is ready
if (typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        window.systemDashboard = new SystemHealthDashboard({
            metricsUrl: '/metrics',
            pollInterval: 2000
        });
    });
}
