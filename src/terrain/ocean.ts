import oceanShaderSource from '../shaders/ocean.wgsl?raw';

export const SEA_LEVEL = 5.0;

// ── Simulation constants ─────────────────────────────────────────────────────
const N           = 128;      // FFT grid resolution (must be power-of-2)
const MESH_N      = 256;      // rendering grid (independent of FFT size)
const TILE_WORLD  = 800.0;    // World-space side length of one tile
const WIND_SPEED  = 20.0;     // m/s
const WIND_DIR    = [1.0, 0.8] as const;
const GRAVITY     = 9.81;
const PHILLIPS_A  = 80.0;     // Wave amplitude scale — 1/N² IFFT normalization shrinks output ~16000×

// ── Phillips spectrum ────────────────────────────────────────────────────────
function phillipsSpectrum(kx: number, kz: number): number {
  const k2 = kx * kx + kz * kz;
  if (k2 < 1e-8) return 0.0;

  const L    = (WIND_SPEED * WIND_SPEED) / GRAVITY;
  const kLen = Math.sqrt(k2);
  const kL   = kLen * L;

  const wLen  = Math.sqrt(WIND_DIR[0] ** 2 + WIND_DIR[1] ** 2);
  const kdotw = (kx * WIND_DIR[0] + kz * WIND_DIR[1]) / (kLen * wLen);

  if (kdotw < 0) return 0.0; // suppress back-propagating waves

  const l = 0.001 * L; // small-wave suppression scale
  return (
    PHILLIPS_A *
    Math.exp(-1.0 / (kL * kL)) /
    (k2 * k2) *
    Math.pow(kdotw, 6) *      // higher exponent = narrower directional spread
    Math.exp(-k2 * l * l)
  );
}

// ── Box-Muller Gaussian sampler ──────────────────────────────────────────────
function gaussRand(rng: () => number): [number, number] {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  const r  = Math.sqrt(-2.0 * Math.log(u1));
  const t  = 2.0 * Math.PI * u2;
  return [r * Math.cos(t), r * Math.sin(t)];
}

// ── Seeded PRNG (mulberry32) ─────────────────────────────────────────────────
function mulberry32(seed: number): () => number {
  return () => {
    seed += 0x6d2b79f5;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Ocean class ──────────────────────────────────────────────────────────────
export class Ocean {
  private pipeline!:     GPURenderPipeline;
  private bindGroup!:    GPUBindGroup;
  private oceanTex!:     GPUTexture;
  private oceanTexView!: GPUTextureView;

  // Precomputed spectrum data
  private h0Re!:  Float32Array; // h0(k).real
  private h0Im!:  Float32Array; // h0(k).imag
  private h0cRe!: Float32Array; // conj(h0(-k)).real
  private h0cIm!: Float32Array; // conj(h0(-k)).imag
  private omega!: Float32Array; // dispersion ω(k)
  private kx!:    Float32Array; // wave-vector x component
  private kz!:    Float32Array; // wave-vector z component

  // Per-frame work buffers (reused to avoid GC pressure)
  private htRe!: Float32Array;
  private htIm!: Float32Array;
  private sxRe!: Float32Array;
  private sxIm!: Float32Array;
  private szRe!: Float32Array;
  private szIm!: Float32Array;

  // rgba32float packed texture: r=height, g=slope_x, b=slope_z, a=0
  private texData!: Float32Array;
  private pipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private globalsBuffer: GPUBuffer,
    private sampleCount: number = 4,
  ) { }

  async init(): Promise<void> {
    this.precomputeSpectrum();
    this.createTexture();
    await this.createPipeline();
  }

  // ── Precompute h0(k) from Phillips spectrum ────────────────────────────────
  private precomputeSpectrum(): void {
    const rng = mulberry32(42);
    const sz  = N * N;

    this.h0Re  = new Float32Array(sz);
    this.h0Im  = new Float32Array(sz);
    this.h0cRe = new Float32Array(sz);
    this.h0cIm = new Float32Array(sz);
    this.omega = new Float32Array(sz);
    this.kx    = new Float32Array(sz);
    this.kz    = new Float32Array(sz);

    this.htRe  = new Float32Array(sz);
    this.htIm  = new Float32Array(sz);
    this.sxRe  = new Float32Array(sz);
    this.sxIm  = new Float32Array(sz);
    this.szRe  = new Float32Array(sz);
    this.szIm  = new Float32Array(sz);
    this.texData = new Float32Array(sz * 4);

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const idx = r * N + c;

        // Centre the frequency grid: [0,N) → [-N/2, N/2)
        const nr  = r < N / 2 ? r : r - N;
        const nc  = c < N / 2 ? c : c - N;
        const kvx = (2 * Math.PI * nc) / TILE_WORLD;
        const kvz = (2 * Math.PI * nr) / TILE_WORLD;
        this.kx[idx] = kvx;
        this.kz[idx] = kvz;

        const kMag       = Math.sqrt(kvx * kvx + kvz * kvz);
        this.omega[idx]  = Math.sqrt(GRAVITY * Math.max(kMag, 1e-4));

        // h0(k) = (1/√2)(ξr + iξi)√P(k)
        const sqrtPk     = Math.sqrt(phillipsSpectrum(kvx, kvz));
        const [g1r, g1i] = gaussRand(rng);
        this.h0Re[idx]   = (1 / Math.SQRT2) * g1r * sqrtPk;
        this.h0Im[idx]   = (1 / Math.SQRT2) * g1i * sqrtPk;

        // conj(h0(-k)) — independent random draw
        const sqrtPnk    = Math.sqrt(phillipsSpectrum(-kvx, -kvz));
        const [g2r, g2i] = gaussRand(rng);
        this.h0cRe[idx]  =  (1 / Math.SQRT2) * g2r * sqrtPnk;
        this.h0cIm[idx]  = -(1 / Math.SQRT2) * g2i * sqrtPnk;
      }
    }
  }

  // ── GPU texture (rgba32float: height, slope_x, slope_z, _) ────────────────
  private createTexture(): void {
    this.oceanTex = this.device.createTexture({
      label:  'Ocean Data Texture',
      size:   { width: N, height: N },
      format: 'rgba32float',
      usage:  GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.oceanTexView = this.oceanTex.createView();
  }

  // ── Render pipeline ────────────────────────────────────────────────────────
  private async createPipeline(): Promise<void> {
    const module = this.device.createShaderModule({
      label: 'Ocean Shader',
      code: oceanShaderSource,
    });

    const bgl = this.device.createBindGroupLayout({
      label:   'Ocean BGL',
      entries: [
        {
          binding:    0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer:     { type: 'uniform' },
        },
        {
          binding:    1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          texture:    { sampleType: 'unfilterable-float', viewDimension: '2d' },
        },
      ],
    });

    this.pipeline = await this.device.createRenderPipelineAsync({
      label:  'Ocean Pipeline',
      layout: this.device.createPipelineLayout({
        label:           'Ocean Layout',
        bindGroupLayouts: [bgl],
      }),
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets:    [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: { count: this.sampleCount },
    });

    this.bindGroup = this.device.createBindGroup({
      label:   'Ocean Bind Group',
      layout:  bgl,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
        { binding: 1, resource: this.oceanTexView },
      ],
    });
  }

  // ── Per-frame: evolve spectrum → IFFT → upload texture ────────────────────
  update(time: number): void {
    const sz = N * N;

    for (let i = 0; i < sz; i++) {
      const w  = this.omega[i] * time;
      const cw = Math.cos(w);
      const sw = Math.sin(w);

      // h(k,t) = h0(k)·e^(iωt) + conj(h0(-k))·e^(-iωt)
      const e1r = this.h0Re[i] * cw - this.h0Im[i] * sw;
      const e1i = this.h0Re[i] * sw + this.h0Im[i] * cw;

      const e2r = this.h0cRe[i] * cw + this.h0cIm[i] * sw;
      const e2i = this.h0cIm[i] * cw - this.h0cRe[i] * sw;

      this.htRe[i] = e1r + e2r;
      this.htIm[i] = e1i + e2i;

      // slope_x(k) = i·kx·h(k,t)  →  multiply by i: (a+ib)·i = -b + ia
      const kvx    = this.kx[i];
      this.sxRe[i] = -kvx * this.htIm[i];
      this.sxIm[i] =  kvx * this.htRe[i];

      // slope_z(k) = i·kz·h(k,t)
      const kvz    = this.kz[i];
      this.szRe[i] = -kvz * this.htIm[i];
      this.szIm[i] =  kvz * this.htRe[i];
    }

    this.ifft2d(this.htRe, this.htIm);
    this.ifft2d(this.sxRe, this.sxIm);
    this.ifft2d(this.szRe, this.szIm);

    // Pack into rgba32float texture
    for (let i = 0; i < sz; i++) {
      this.texData[i * 4 + 0] = this.htRe[i]; // height
      this.texData[i * 4 + 1] = this.sxRe[i]; // ∂H/∂x
      this.texData[i * 4 + 2] = this.szRe[i]; // ∂H/∂z
      this.texData[i * 4 + 3] = 0.0;
    }

    this.device.queue.writeTexture(
      { texture: this.oceanTex },
      this.texData,
      { bytesPerRow: N * 16 }, // 4 floats × 4 bytes × N pixels
      { width: N, height: N },
    );
  }

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(MESH_N * MESH_N * 6); // MESH_N×MESH_N quads, 2 triangles each
  }

  // ── 2-D IFFT (separable row + column passes) ───────────────────────────────
  private ifft2d(re: Float32Array, im: Float32Array): void {
    for (let r = 0; r < N; r++) this.fft1d(re, im, r * N, 1,  N, true);
    for (let c = 0; c < N; c++) this.fft1d(re, im, c,     N,  N, true);
  }

  // ── In-place radix-2 Cooley-Tukey FFT/IFFT ────────────────────────────────
  private fft1d(
    re: Float32Array, im: Float32Array,
    offset: number, stride: number, n: number, invert: boolean,
  ): void {
    // Bit-reversal permutation
    let j = 0;
    for (let i = 1; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        const ti = offset + i * stride, tj = offset + j * stride;
        let t = re[ti]; re[ti] = re[tj]; re[tj] = t;
            t = im[ti]; im[ti] = im[tj]; im[tj] = t;
      }
    }

    // Butterfly passes
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (invert ? 1 : -1) * (2 * Math.PI) / len;
      const wr  = Math.cos(ang);
      const wi  = Math.sin(ang);

      for (let i = 0; i < n; i += len) {
        let cr = 1.0, ci = 0.0;
        const half = len >> 1;
        for (let jj = 0; jj < half; jj++) {
          const ui = offset + (i + jj)        * stride;
          const vi = offset + (i + jj + half) * stride;

          const ur = re[ui], uim = im[ui];
          const vr = re[vi], vim = im[vi];

          const tvr = vr * cr - vim * ci;
          const tvi = vr * ci + vim * cr;

          re[ui] = ur + tvr;  im[ui] = uim + tvi;
          re[vi] = ur - tvr;  im[vi] = uim - tvi;

          const ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr;
          cr = ncr;
        }
      }
    }

    // Normalise for inverse transform
    if (invert) {
      const inv = 1.0 / n;
      for (let i = 0; i < n; i++) {
        re[offset + i * stride] *= inv;
        im[offset + i * stride] *= inv;
      }
    }
  }
}
