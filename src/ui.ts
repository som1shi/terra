function srgbChannelToLinear255(c: number): number {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

function linearToSrgb255Channel(x: number): number {
  const c = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  return Math.min(255, Math.max(0, Math.round(c * 255)));
}

function rgb255ToOklab(r: number, g: number, b: number): [number, number, number] {
  const rl = srgbChannelToLinear255(r);
  const gl = srgbChannelToLinear255(g);
  const bl = srgbChannelToLinear255(b);
  const l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl;
  const m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl;
  const s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl;
  const l_ = Math.cbrt(l);
  const m_ = Math.cbrt(m);
  const s_ = Math.cbrt(s);
  return [
    0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
    1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
    0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
  ];
}

function oklabToRgb255(L: number, a: number, bCh: number): [number, number, number] {
  const l_ = L + 0.3963377774 * a + 0.2158037573 * bCh;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * bCh;
  const s_ = L - 0.0894841775 * a - 1.2914855480 * bCh;
  const l = l_ * l_ * l_;
  const m = m_ * m_ * m_;
  const s = s_ * s_ * s_;
  const r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
  const g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
  const bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
  return [linearToSrgb255Channel(r), linearToSrgb255Channel(g), linearToSrgb255Channel(bl)];
}

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
    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;
    const R = cx * 0.74;
    const lineW = cx * 0.145;

    ctx.clearRect(0, 0, W, H);

    const SEG = 1024;
    const lw = lineW * 2 + 1.35;
    ctx.lineCap = 'butt';
    for (let i = 0; i < SEG; i++) {
      const a0 = (i / SEG) * Math.PI * 2 - Math.PI / 2;
      const a1 = ((i + 1) / SEG) * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx, cy, R, a0, a1);
      ctx.strokeStyle = this.todColor((i + 0.5) / SEG);
      ctx.lineWidth = lw;
      ctx.stroke();
    }

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(0,4,18,0.60)';
    ctx.beginPath();
    ctx.arc(cx, cy, R - lineW + 1, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, R + lineW - 1, 0, Math.PI * 2);
    ctx.stroke();

    const tickR = R + lineW + 7;
    for (let h = 0; h < 12; h++) {
      const a = (h / 12) * Math.PI * 2 - Math.PI / 2;
      ctx.beginPath();
      ctx.arc(cx + Math.cos(a) * tickR, cy + Math.sin(a) * tickR, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(130, 185, 255, 0.28)';
      ctx.fill();
    }

    const kAngle = this.todValue * Math.PI * 2 - Math.PI / 2;
    const kx = cx + Math.cos(kAngle) * R;
    const ky = cy + Math.sin(kAngle) * R;

    const halo = ctx.createRadialGradient(kx, ky, 0, kx, ky, 22);
    halo.addColorStop(0, 'rgba(210, 232, 255, 0.22)');
    halo.addColorStop(1, 'rgba(210, 232, 255, 0)');
    ctx.beginPath();
    ctx.arc(kx, ky, 22, 0, Math.PI * 2);
    ctx.fillStyle = halo;
    ctx.fill();

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

    ctx.beginPath();
    ctx.arc(kx, ky, 9, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(190, 222, 255, 0.60)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  private todColor(t: number): string {
    const midnight = [10, 20, 76] as const;
    const indigo = [36, 44, 138] as const;
    const violet = [82, 56, 178] as const;
    const magentaRose = [168, 58, 148] as const;
    const coral = [228, 88, 68] as const;
    const sunset = [236, 108, 38] as const;
    const amber = [252, 168, 62] as const;
    const honey = [255, 228, 158] as const;
    const seaFoam = [158, 232, 212] as const;
    const aqua = [72, 188, 238] as const;
    const sky = [38, 148, 248] as const;

    const stops: [number, readonly [number, number, number]][] = [
      [0.00, midnight],
      [0.045, indigo],
      [0.10, violet],
      [0.155, magentaRose],
      [0.205, coral],
      [0.25, sunset],
      [0.295, amber],
      [0.34, honey],
      [0.39, seaFoam],
      [0.445, aqua],
      [0.5, sky],
      [0.555, aqua],
      [0.61, seaFoam],
      [0.66, honey],
      [0.705, amber],
      [0.75, sunset],
      [0.795, coral],
      [0.845, magentaRose],
      [0.90, violet],
      [0.955, indigo],
      [1.00, midnight],
    ];

    let x = t % 1;
    if (x < 0) x += 1;

    let lo = stops[0];
    let hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
      if (x >= stops[i][0] && x <= stops[i + 1][0]) {
        lo = stops[i];
        hi = stops[i + 1];
        break;
      }
    }

    const span = hi[0] - lo[0];
    const u = span > 1e-9 ? (x - lo[0]) / span : 0;

    const lab0 = rgb255ToOklab(lo[1][0], lo[1][1], lo[1][2]);
    const lab1 = rgb255ToOklab(hi[1][0], hi[1][1], hi[1][2]);
    const L = lab0[0] + u * (lab1[0] - lab0[0]);
    const a = lab0[1] + u * (lab1[1] - lab0[1]);
    const bLab = lab0[2] + u * (lab1[2] - lab0[2]);
    const [r, g, b] = oklabToRgb255(L, a, bLab);
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
