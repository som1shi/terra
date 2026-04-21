import { Renderer } from './renderer';
import { UI } from './ui';
import { CLIMATE_PRESETS, type ClimatePreset } from './atmosphere/atmosphere';

const CLIMATE_LABELS: Record<ClimatePreset, string> = {
  temperate: '🌤 Temperate',
  arid:      '☀️ Arid',
  tropical:  '🌴 Tropical',
  arctic:    '❄️ Arctic',
  stormy:    '⛈ Stormy',
};

// Fog density when "on" — matches the original hardcoded shader default of 0.00005
const FOG_ON_DENSITY  = 0.00005;
const FOG_OFF_DENSITY = 0.0;

async function main(): Promise<void> {
  const ui = new UI();

  if (!navigator.gpu) {
    ui.showError(
      'WebGPU is not supported in this browser. ' +
      'Please use Chrome 113+, Edge 113+, or another WebGPU-capable browser.'
    );
    return;
  }

  try {
    ui.setStatus('Initializing WebGPU...', 5);

    let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      adapter = await navigator.gpu.requestAdapter();
    }

    if (!adapter) {
      ui.showError(
        'No suitable WebGPU adapter found. ' +
        'Your GPU may not support the required features.'
      );
      return;
    }

    try {
      const adapterInfo = await (adapter as unknown as { requestAdapterInfo(): Promise<{ vendor: string; device: string }> }).requestAdapterInfo();
      console.log('WebGPU Adapter:', adapterInfo.vendor, adapterInfo.device);
    } catch {
      console.log('WebGPU adapter info unavailable');
    }

    const requiredFeatures: GPUFeatureName[] = [];

    if (adapter.features.has('float32-filterable')) {
      requiredFeatures.push('float32-filterable');
    }

    const device = await adapter.requestDevice({ requiredFeatures });

    device.lost.then((info) => {
      console.error('WebGPU device lost:', info.message, info.reason);
      if (info.reason !== 'destroyed') {
        ui.showError(`WebGPU device lost: ${info.message}. Please refresh the page.`);
      }
    });

    ui.setStatus('Configuring canvas...', 10);

    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;

    const context = canvas.getContext('webgpu');
    if (!context) {
      ui.showError('Failed to get WebGPU canvas context.');
      return;
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    window.addEventListener('resize', () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
      context.configure({ device, format, alphaMode: 'opaque' });
      renderer.onResize(canvas.width, canvas.height);
    });

    const renderer = new Renderer(device, context, format, canvas, ui);
    await renderer.init();

    ui.hideLoading();
    ui.showUI();
    ui.onTimeOfDayChange((tod) => renderer.setTimeOfDay(tod));

    // ── State ─────────────────────────────────────────────────────────────────
    let fogOn     = true;
    let presetIdx = 0;

    renderer.setFogDensity(FOG_ON_DENSITY);
    renderer.setClimatePreset(CLIMATE_PRESETS[presetIdx]);

    // ── HUD toast (bottom-centre, fades out) ──────────────────────────────────
    const hud = document.createElement('div');
    hud.style.cssText = [
      'position:fixed', 'bottom:20px', 'left:50%', 'transform:translateX(-50%)',
      'background:rgba(0,0,0,0.55)', 'color:#dde',
      'font:12px/1 monospace', 'padding:6px 16px',
      'border-radius:20px', 'pointer-events:none', 'z-index:200',
      'white-space:nowrap', 'opacity:0', 'transition:opacity 0.3s',
    ].join(';');
    document.body.appendChild(hud);

    let hudTimer: ReturnType<typeof setTimeout> | null = null;
    function showHUD(msg: string): void {
      hud.textContent   = msg;
      hud.style.opacity = '1';
      if (hudTimer) clearTimeout(hudTimer);
      hudTimer = setTimeout(() => { hud.style.opacity = '0'; }, 1800);
    }


    // ── Keyboard handler ──────────────────────────────────────────────────────
    window.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      switch (e.key.toLowerCase()) {
        case 'f': {
          fogOn = !fogOn;
          renderer.setFogDensity(fogOn ? FOG_ON_DENSITY : FOG_OFF_DENSITY);
          showHUD(`Fog  ${fogOn ? 'ON  🌫' : 'OFF'}`);
          break;
        }
        case 'c': {
          presetIdx = (presetIdx + 1) % CLIMATE_PRESETS.length;
          const preset = CLIMATE_PRESETS[presetIdx];
          renderer.setClimatePreset(preset);
          showHUD(`Climate  ${CLIMATE_LABELS[preset]}`);
          break;
        }
      }
    });

    // ── Frame loop ────────────────────────────────────────────────────────────
    let lastTime   = performance.now();
    let frameCount = 0;
    let fpsAccum   = 0;

    function frame(timestamp: number): void {
      const dt = Math.min((timestamp - lastTime) / 1000, 0.05);
      lastTime = timestamp;

      frameCount++;
      fpsAccum += dt;

      if (fpsAccum >= 1.0) {
        const fps     = Math.round(frameCount / fpsAccum);
        const frameMs = ((fpsAccum / frameCount) * 1000).toFixed(1);
        ui.updateStats(fps, parseFloat(frameMs), renderer.getCameraPosition());
        frameCount = 0;
        fpsAccum   = 0;
      }

      renderer.render(timestamp / 1000, dt);
      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);

  } catch (err) {
    console.error('Fatal error during initialization:', err);
    ui.showError(
      `Initialization failed: ${err instanceof Error ? err.message : String(err)}`
    );
  }
}

main();
