// WebSocket connection
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateCharts(data);
    updateMetrics(data);
};

// Chart initialization
function initializeCharts() {
    // Win Rate Chart
    const winRateCtx = document.getElementById('winRateChart').getContext('2d');
    window.winRateChart = new Chart(winRateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'vs MCTS',
                data: [],
                borderColor: '#2ecc71',
                fill: false
            },
            {
                label: 'vs CFR',
                data: [],
                borderColor: '#e74c3c',
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // Profit Distribution Chart
    const profitCtx = document.getElementById('profitChart').getContext('2d');
    window.profitChart = new Chart(profitCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'GT-DQN',
                data: [],
                borderColor: '#2ecc71'
            },
            {
                label: 'MCTS',
                data: [],
                borderColor: '#e74c3c'
            },
            {
                label: 'CFR',
                data: [],
                borderColor: '#3498db'
            }]
        }
    });
}

// Update charts with new data
function updateCharts(data) {
    // Update win rate chart
    winRateChart.data.labels = data.episodes;
    winRateChart.data.datasets[0].data = data.gt_dqn_vs_mcts_winrate;
    winRateChart.data.datasets[1].data = data.gt_dqn_vs_cfr_winrate;
    winRateChart.update();
    
    // Update profit chart
    profitChart.data.labels = data.episodes;
    profitChart.data.datasets[0].data = data.gt_dqn_profits;
    profitChart.data.datasets[1].data = data.mcts_profits;
    profitChart.data.datasets[2].data = data.cfr_profits;
    profitChart.update();
}

// Update metrics display
function updateMetrics(data) {
    const lastEpisode = data.episodes[data.episodes.length - 1];
    const mctsWinRate = data.gt_dqn_vs_mcts_winrate[data.gt_dqn_vs_mcts_winrate.length - 1];
    const cfrWinRate = data.gt_dqn_vs_cfr_winrate[data.gt_dqn_vs_cfr_winrate.length - 1];
    
    document.getElementById('mctsWinRate').textContent = `${(mctsWinRate * 100).toFixed(1)}%`;
    document.getElementById('cfrWinRate').textContent = `${(cfrWinRate * 100).toFixed(1)}%`;
    document.getElementById('episode').textContent = lastEpisode;
}

// Initialize on load
document.addEventListener('DOMContentLoaded', initializeCharts);
