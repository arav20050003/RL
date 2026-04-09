// charts.js
document.addEventListener('DOMContentLoaded', () => {
  // Wait a tiny bit for fonts to load before drawing to ensure correct dimensions
  setTimeout(initCharts, 200);
});

function initCharts() {
  // Chart.js global defaults for Dark Theme
  Chart.defaults.color = '#8899aa';
  Chart.defaults.borderColor = '#1a2332';
  Chart.defaults.font.family = "'Space Mono', monospace";
  Chart.defaults.font.size = 11;

  Chart.defaults.plugins.legend.labels.color = '#8899aa';
  Chart.defaults.plugins.legend.labels.padding = 16;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0d1420'; // bg-card
  Chart.defaults.plugins.tooltip.borderColor = '#00D4FF'; // cyan
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = '#00D4FF';
  Chart.defaults.plugins.tooltip.bodyColor = '#ffffff';
  Chart.defaults.plugins.tooltip.padding = 12;

  drawPhase1Chart();
  drawPhase2FrequentShort();
  drawPhase2InfrequentLong();
  drawPhase2StressTest();
  drawOracleDonuts();
}

function drawPhase1Chart() {
  const ctx = document.getElementById('chart-phase1');
  if (!ctx) return;

  const data = PROJECT_DATA.phase1.policies;
  const labels = data.map(d => d.name);
  const paperMeans = data.map(d => d.paper_mean);
  const ourMeans = data.map(d => d.our_mean);
  // Error bars would normally require a plugin or scatter overlay, 
  // keeping it simple per standard Chart.js bar chart for now to guarantee functionality
  // without relying on external non-standard plugins.

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Paper Result',
          data: paperMeans,
          backgroundColor: 'rgba(0, 212, 255, 0.3)',
          borderColor: '#00D4FF',
          borderWidth: 1,
          barPercentage: 0.8,
          categoryPercentage: 0.8
        },
        {
          label: 'Our Implementation',
          data: ourMeans,
          backgroundColor: '#00D4FF',
          borderColor: '#00D4FF',
          borderWidth: 1,
          barPercentage: 0.8,
          categoryPercentage: 0.8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: '#1a2a3a' }
        },
        x: {
          grid: { display: false }
        }
      },
      plugins: {
        legend: { position: 'top', align: 'end' }
      }
    }
  });
}

function getPhase2Colors() {
  return [
    '#4488aa', // (s,Q)
    '#00D4FF', // PPO Blind
    '#FF4444', // PPO Disrpt-Aware
    '#00E676'  // PPO LLM-Aug
  ];
}

function drawPhase2FrequentShort() {
  const ctx = document.getElementById('chart-fs-bar');
  const radarCtx = document.getElementById('chart-fs-radar');
  if (!ctx || !radarCtx) return;

  const data = PROJECT_DATA.phase2.frequent_short;
  const labels = data.map(d => d.name);
  const profits = data.map(d => d.profit);
  const colors = getPhase2Colors();

  // Bar Chart
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Average Profit',
        data: profits,
        backgroundColor: colors,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true }
      },
      plugins: {
        legend: { display: false }
      }
    }
  });

  // Radar Chart
  new Chart(radarCtx, {
    type: 'radar',
    data: {
      labels: ['Profit (Norm)', 'Service Level', 'Low Stockout', '-Disruption Cost'],
      datasets: data.map((d, i) => {
        // Normalize for visual scale
        const prof = Math.max(0, d.profit / 1100);
        const sl = d.service;
        const ls = 1 - d.stockout;
        const dc = 1 - (d.disrpt_cost / 200); 
        
        // Convert hex to rgba for fill
        let fill;
        if(i===0) fill = 'rgba(68,136,170,0.2)';
        if(i===1) fill = 'rgba(0,212,255,0.2)';
        if(i===2) fill = 'rgba(255,68,68,0.2)';
        if(i===3) fill = 'rgba(0,230,118,0.2)';

        return {
          label: d.name,
          data: [prof, sl, ls, dc],
          borderColor: colors[i],
          backgroundColor: fill,
          borderWidth: 2,
          pointBackgroundColor: colors[i]
        };
      })
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: '#1a2a3a' },
          grid: { color: '#1a2a3a' },
          pointLabels: { color: '#E8EDF2', font: { family: "'Space Mono', monospace" } },
          ticks: { display: false, min: 0, max: 1 }
        }
      },
      plugins: {
        legend: { position: 'bottom' }
      }
    }
  });
}

function drawPhase2InfrequentLong() {
  const ctx = document.getElementById('chart-il-bar');
  if (!ctx) return;
  const data = PROJECT_DATA.phase2.infrequent_long;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.name),
      datasets: [{
        label: 'Average Profit',
        data: data.map(d => d.profit),
        backgroundColor: getPhase2Colors(),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });
}

function drawPhase2StressTest() {
  const ctx = document.getElementById('chart-st-bar');
  if (!ctx) return;
  const data = PROJECT_DATA.phase2.stress_test;
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.map(d => d.name),
      datasets: [{
        label: 'Average Profit',
        data: data.map(d => d.profit),
        backgroundColor: getPhase2Colors(),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: { y: { suggestedMax: 0 } }, // Negative bars
      plugins: { legend: { display: false } }
    }
  });
}

function drawOracleDonuts() {
  const tprCtx = document.getElementById('chart-donut-tpr');
  const fprCtx = document.getElementById('chart-donut-fpr');
  if (!tprCtx || !fprCtx) return;

  const od = PROJECT_DATA.oracle_news;

  new Chart(tprCtx, {
    type: 'doughnut',
    data: {
      labels: ['Detected', 'Missed'],
      datasets: [{
        data: [od.tpr, 100 - od.tpr],
        backgroundColor: ['#00E676', '#1a2a3a'],
        borderWidth: 0,
        cutout: '80%'
      }]
    },
    options: { responsive: true, plugins: { legend: { display: false }, tooltip: {enabled: false} } }
  });

  new Chart(fprCtx, {
    type: 'doughnut',
    data: {
      labels: ['False Positive', 'True Negative'],
      datasets: [{
        data: [od.fpr, 100 - od.fpr],
        backgroundColor: ['#FFB347', '#1a2a3a'],
        borderWidth: 0,
        cutout: '80%'
      }]
    },
    options: { responsive: true, plugins: { legend: { display: false }, tooltip: {enabled: false} } }
  });
}
