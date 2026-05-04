export class UI {
  private loadingOverlay: HTMLElement;
  private loadingBar: HTMLElement;
  private loadingStatus: HTMLElement;
  private errorOverlay: HTMLElement;
  private errorMessage: HTMLElement;
  private uiPanel: HTMLElement;
  private stats: HTMLElement;
  private controlsHint: HTMLElement;
  private fpsEl: HTMLElement;
  private frameTimeEl: HTMLElement;
  private camPosEl: HTMLElement;
  private todDial!: HTMLCanvasElement;
  private todDisplay!: HTMLElement;
  private todCallback: ((value: number) => void) | null = null;
  private todValue = 0.45;
  private dragging = false;

  constructor() {
    this.loadingOverlay = document.getElementById('loading-overlay')!;
    this.loadingBar = document.getElementById('loading-bar')!;
    this.loadingStatus = document.getElementById('loading-status')!;
    this.errorOverlay = document.getElementById('error-overlay')!;
    this.errorMessage = document.getElementById('error-message')!;
    this.uiPanel = document.getElementById('ui-panel')!;
    this.stats = document.getElementById('stats')!;
    this.controlsHint = document.getElementById('controls-hint')!;
    this.fpsEl = document.getElementById('fps')!;
    this.frameTimeEl = document.getElementById('frame-time')!;
    this.camPosEl = document.getElementById('cam-pos')!;
    this.todDial = document.getElementById('tod-dial') as HTMLCanvasElement;
    this.todDisplay = document.getElementById('tod-display')!;

    this.initDial();
  }

  private initDial(): void {
    const onPointerDown = (e: PointerEvent) => {
      this.dragging = true;
      this.todDial.setPointerCapture(e.pointerId);
      this.handleDialPointer(e);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (this.dragging) this.handleDialPointer(e);
    };
    const onPointerUp = () => { this.dragging = false; };

    this.todDial.addEventListener('pointerdown', onPointerDown);
    this.todDial.addEventListener('pointermove', onPointerMove);
    this.todDial.addEventListener('pointerup', onPointerUp);

    this.updateTodDisplay(this.todValue);
    this.drawDial();
  }

  private handleDialPointer(e: PointerEvent): void {
    const rect = this.todDial.getBoundingClientRect();
    const dx = e.clientX - (rect.left + rect.width / 2);
    const dy = e.clientY - (rect.top + rect.height / 2);
    // atan2(dx, -dy): angle from 12 o'clock going clockwise
    const angle = Math.atan2(dx, -dy);
    const val = ((angle / (Math.PI * 2)) + 1) % 1;
    this.todValue = val;
    this.updateTodDisplay(val);
    this.drawDial();
    if (this.todCallback) this.todCallback(val);
  }

  private drawDial(): void {
    const canvas = this.todDial;
    const ctx = canvas.getContext('2d')!;
    const W = canvas.width;   // internal pixels (296)
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const R = cx * 0.74;      // ring center radius
    const lineW = cx * 0.145; // ring half-width

    ctx.clearRect(0, 0, W, H);

    // Gradient ring — 360 arc segments blended together
    const SEGS = 360;
    for (let i = 0; i < SEGS; i++) {
      const t = i / SEGS;
      const a0 = t * Math.PI * 2 - Math.PI / 2;
      const a1 = (i + 1.5) / SEGS * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, R, a0, a1);
      ctx.strokeStyle = this.todColor(t);
      ctx.lineWidth = lineW * 2;
      ctx.lineCap = 'butt';
      ctx.stroke();
    }

    // Subtle dark edge lines on inner/outer ring rim
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(0,4,18,0.60)';
    ctx.beginPath();
    ctx.arc(cx, cy, R - lineW + 1, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, R + lineW - 1, 0, Math.PI * 2);
    ctx.stroke();

    // Hour tick dots just outside the ring (12 per revolution)
    const tickR = R + lineW + 7;
    for (let h = 0; h < 12; h++) {
      const a = (h / 12) * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx + Math.cos(a) * tickR, cy + Math.sin(a) * tickR, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(130, 185, 255, 0.28)';
      ctx.fill();
    }

    // Knob position on the ring
    const kAngle = this.todValue * Math.PI * 2 - Math.PI / 2;
    const kx = cx + Math.cos(kAngle) * R;
    const ky = cy + Math.sin(kAngle) * R;

    // Soft glow halo behind knob
    const halo = ctx.createRadialGradient(kx, ky, 0, kx, ky, 22);
    halo.addColorStop(0, 'rgba(210, 232, 255, 0.22)');
    halo.addColorStop(1, 'rgba(210, 232, 255, 0)');
    ctx.beginPath();
    ctx.arc(kx, ky, 22, 0, Math.PI * 2);
    ctx.fillStyle = halo;
    ctx.fill();

    // Knob — white circle with drop shadow
    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.60)';
    ctx.shadowBlur = 10;
    ctx.shadowOffsetX = 1;
    ctx.shadowOffsetY = 2;
    ctx.beginPath();
    ctx.arc(kx, ky, 9, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.restore();

    // Knob specular ring
    ctx.beginPath();
    ctx.arc(kx, ky, 9, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(190, 222, 255, 0.60)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Interpolates the same color stops as the original linear gradient
  private todColor(t: number): string {
    const stops: [number, [number, number, number]][] = [
      [0.00, [8,   8,   32]],
      [0.17, [22,  40,  120]],
      [0.36, [216, 118, 48]],
      [0.50, [240, 204, 80]],
      [0.63, [96,  184, 232]],
      [0.82, [22,  40,  120]],
      [1.00, [8,   8,   32]],
    ];
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (t >= stops[i][0] && t <= stops[i + 1][0]) {
        lo = stops[i]; hi = stops[i + 1]; break;
      }
    }
    const f = hi[0] > lo[0] ? (t - lo[0]) / (hi[0] - lo[0]) : 0;
    const r = Math.round(lo[1][0] + f * (hi[1][0] - lo[1][0]));
    const g = Math.round(lo[1][1] + f * (hi[1][1] - lo[1][1]));
    const b = Math.round(lo[1][2] + f * (hi[1][2] - lo[1][2]));
    return `rgb(${r},${g},${b})`;
  }

  setStatus(message: string, progress: number): void {
    this.loadingStatus.textContent = message;
    this.loadingBar.style.width = `${progress}%`;
  }

  showError(message: string): void {
    this.loadingOverlay.style.display = 'none';
    this.errorMessage.innerHTML = message;
    this.errorOverlay.classList.add('visible');
  }

  hideLoading(): void {
    this.loadingOverlay.classList.add('fade-out');
    setTimeout(() => {
      this.loadingOverlay.style.display = 'none';
      (window as unknown as { _stopGrain?: () => void })._stopGrain?.();
    }, 800);
  }

  showUI(): void {
    this.uiPanel.classList.add('visible');
    this.stats.classList.add('visible');
    this.controlsHint.classList.add('visible');
  }

  updateStats(fps: number, frameMs: number, camPos: [number, number, number]): void {
    this.fpsEl.textContent = String(fps);
    this.frameTimeEl.textContent = frameMs.toFixed(1);
    this.camPosEl.textContent =
      `${camPos[0].toFixed(0)}, ${camPos[1].toFixed(0)}, ${camPos[2].toFixed(0)}`;
  }

  onTimeOfDayChange(callback: (value: number) => void): void {
    this.todCallback = callback;
    callback(this.todValue);
  }

  getTimeOfDay(): number {
    return this.todValue;
  }

  private updateTodDisplay(value: number): void {
    const hours = (value * 24) % 24;
    const h = Math.floor(hours);
    const m = Math.floor((hours - h) * 60);
    this.todDisplay.textContent =
      `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
  }
}
