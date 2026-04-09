// animations.js

document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  initObservers();
  initTypewriter();
  initNavbar();
  initTabs();
  initOracleDemo();
});

// 1. Particle System (Supply Chain Network Metaphor)
function initParticles() {
  const canvas = document.getElementById('particles');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  
  let width, height;
  let particles = [];
  
  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  
  window.addEventListener('resize', resize);
  resize();
  
  class Particle {
    constructor() {
      this.x = Math.random() * width;
      this.y = Math.random() * height;
      this.vx = (Math.random() - 0.5) * 0.5;
      this.vy = (Math.random() - 0.5) * 0.5;
      this.radius = Math.random() * 1.5 + 1;
      this.isBright = Math.random() > 0.9;
    }
    
    update() {
      this.x += this.vx;
      this.y += this.vy;
      
      if (this.x < 0 || this.x > width) this.vx *= -1;
      if (this.y < 0 || this.y > height) this.vy *= -1;
    }
    
    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.fillStyle = this.isBright ? '#00D4FF' : 'rgba(0, 212, 255, 0.3)';
      ctx.fill();
    }
  }
  
  for (let i = 0; i < 60; i++) {
    particles.push(new Particle());
  }
  
  function animate() {
    ctx.clearRect(0, 0, width, height);
    
    // Draw lines
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < 120) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.strokeStyle = `rgba(0, 212, 255, ${0.2 * (1 - dist/120)})`;
          ctx.stroke();
        }
      }
    }
    
    // Draw particles
    particles.forEach(p => {
      p.update();
      p.draw();
    });
    
    requestAnimationFrame(animate);
  }
  
  animate();
}

// 2. Intersection Observers (Scroll animations and Count-ups)
function initObservers() {
  // Animate elements up on scroll
  const upObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
      }
    });
  }, { threshold: 0.1 });
  
  document.querySelectorAll('.animate-up, .tl-node').forEach(el => {
    upObserver.observe(el);
  });
  
  // Count up numbers
  const countObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
        entry.target.classList.add('counted');
        countUp(entry.target);
      }
    });
  }, { threshold: 0.3 });
  
  document.querySelectorAll('.count-up').forEach(el => {
    countObserver.observe(el);
  });
}

function countUp(el) {
  const target = parseFloat(el.getAttribute('data-target'));
  const isDecimal = el.getAttribute('data-target').includes('.');
  const duration = 1500;
  let start = null;
  
  function step(timestamp) {
    if (!start) start = timestamp;
    const progress = timestamp - start;
    const ratio = Math.min(progress / duration, 1);
    
    // easeOutQuart
    const ease = 1 - Math.pow(1 - ratio, 4);
    const current = ease * target;
    
    if (isDecimal) {
      el.innerText = current.toFixed(2);
    } else {
      el.innerText = Math.floor(current);
    }
    
    if (progress < duration) {
      window.requestAnimationFrame(step);
    } else {
      el.innerText = isDecimal ? target.toFixed(2) : target;
    }
  }
  
  window.requestAnimationFrame(step);
}

// 3. Typewriter Effect
function initTypewriter() {
  const el = document.getElementById('typewriter');
  if (!el) return;
  
  const text = el.getAttribute('data-text');
  el.innerText = '';
  let i = 0;
  
  // Adding cursor
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  el.parentNode.appendChild(cursor);
  
  setTimeout(() => {
    function type() {
      if (i < text.length) {
        el.innerText += text.charAt(i);
        i++;
        setTimeout(type, 80);
      }
    }
    type();
  }, 1000);
}

// 4. Navbar scroll hiding
function initNavbar() {
  const nav = document.querySelector('nav');
  let lastScrollY = window.scrollY;
  
  window.addEventListener('scroll', () => {
    if (window.scrollY > 100) {
      if (window.scrollY > lastScrollY) {
        nav.classList.add('nav-hidden'); // scrolling down
      } else {
        nav.classList.remove('nav-hidden'); // scrolling up
      }
    } else {
      nav.classList.remove('nav-hidden');
    }
    lastScrollY = window.scrollY;
  });
}

// 5. Tabs (Phase 2)
function initTabs() {
  const btns = document.querySelectorAll('.tab-btn');
  const contents = document.querySelectorAll('.tab-content');
  
  btns.forEach(btn => {
    btn.addEventListener('click', () => {
      // Remove active classes
      btns.forEach(b => b.classList.remove('active'));
      contents.forEach(c => c.classList.remove('active'));
      
      // Add active to clicked
      btn.classList.add('active');
      const target = document.getElementById(btn.getAttribute('data-target'));
      target.classList.add('active');
      
      // Re-trigger chart animation if needed
      // (Chart.js handles some of this out of the box if we update it, 
      // but here we have hidden canvases, so they might need a resize trigger)
      window.dispatchEvent(new Event('resize'));
    });
  });
}

// 6. Oracle Interactive Demo
function initOracleDemo() {
  const btn1 = document.getElementById('btn-sim-1');
  const btn0 = document.getElementById('btn-sim-0');
  const hlBox = document.getElementById('demo-headline');
  const scoreBox = document.getElementById('demo-score');
  const scoreWrap = document.getElementById('demo-score-wrap');
  
  if (!btn1 || !btn0) return;
  
  const d1_headline = PROJECT_DATA.oracle_news.disruption_headlines[0];
  const d0_headline = PROJECT_DATA.oracle_news.normal_headlines[0];
  const kw = PROJECT_DATA.oracle_news.keywords;
  
  function runDemo(isDisruption) {
    btn1.classList.remove('active');
    btn0.classList.remove('active');
    if (isDisruption) btn1.classList.add('active');
    else btn0.classList.add('active');
    
    scoreWrap.classList.remove('visible');
    hlBox.innerHTML = '';
    
    const text = isDisruption ? d1_headline : d0_headline;
    let i = 0;
    
    // Fast typewriter
    function type() {
      if (i < text.length) {
        hlBox.innerHTML += text.charAt(i);
        i++;
        setTimeout(type, 30);
      } else {
        setTimeout(() => highlightAndScore(text, isDisruption), 300);
      }
    }
    type();
  }
  
  function highlightAndScore(text, isDisruption) {
    // 1. Highlight keywords
    let html = text;
    Object.keys(kw).forEach(word => {
      const regex = new RegExp(`(${word})`, 'gi');
      html = html.replace(regex, `<span class="highlight-kw">$1</span>`);
    });
    hlBox.innerHTML = html;
    
    // 2. Compute/Display score
    let targetScore = isDisruption ? 0.876 : 0.127;
    // Add bit of fake random noise like the real model
    targetScore += (Math.random() * 0.05 - 0.02);
    targetScore = Math.max(0, Math.min(1, targetScore));
    
    scoreBox.setAttribute('data-target', targetScore.toString());
    scoreBox.innerText = '0.00';
    scoreWrap.classList.add('visible');
    
    // Reuse count up logic but slightly tailored
    const duration = 1000;
    let start = null;
    function step(timestamp) {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const ratio = Math.min(progress / duration, 1);
      const ease = 1 - Math.pow(1 - ratio, 4);
      scoreBox.innerText = (ease * targetScore).toFixed(3);
      if (progress < duration) window.requestAnimationFrame(step);
      else scoreBox.innerText = targetScore.toFixed(3);
    }
    window.requestAnimationFrame(step);
  }
  
  btn1.addEventListener('click', () => runDemo(true));
  btn0.addEventListener('click', () => runDemo(false));
}
