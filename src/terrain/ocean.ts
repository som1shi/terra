import oceanShaderSource   from '../shaders/ocean.wgsl?raw';
import oceanFftShaderSource from '../shaders/ocean_fft.wgsl?raw';

export const SEA_LEVEL = 5.0;

const N          = 128;
const MESH_N     = 256;
const TILE_WORLD = 800.0;
const WIND_SPEED = 20.0;
const WIND_DIR   = [1.0, 0.8] as const;
const GRAVITY    = 9.81;
const PHILLIPS_A = 80.0;

function phillipsSpectrum(kx: number, kz: number): number {
  const k2 = kx * kx + kz * kz;
  if (k2 < 1e-8) return 0.0;
  const L    = (WIND_SPEED * WIND_SPEED) / GRAVITY;
  const kLen = Math.sqrt(k2);
  const kL   = kLen * L;
  const wLen = Math.sqrt(WIND_DIR[0] ** 2 + WIND_DIR[1] ** 2);
  const kdotw = (kx * WIND_DIR[0] + kz * WIND_DIR[1]) / (kLen * wLen);
  if (kdotw < 0) return 0.0;
  const l = 0.001 * L;
  return PHILLIPS_A * Math.exp(-1.0 / (kL * kL)) / (k2 * k2) *
         Math.pow(kdotw, 6) * Math.exp(-k2 * l * l);
}

function gaussRand(rng: () => number): [number, number] {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  const r  = Math.sqrt(-2.0 * Math.log(u1));
  const t  = 2.0 * Math.PI * u2;
  return [r * Math.cos(t), r * Math.sin(t)];
}

function mulberry32(seed: number): () => number {
  return () => {
    seed += 0x6d2b79f5;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Ocean {
  private renderPipeline!: GPURenderPipeline;
  private renderBindGroup!: GPUBindGroup;
  private oceanTex!: GPUTexture;
  private oceanTexView!: GPUTextureView;

  private h0Buf!: GPUBuffer;
  private freqBuf!: GPUBuffer;
  private specBuf!: GPUBuffer;
  private spectrumPipeline!: GPUComputePipeline;
  private fftRowsPipeline!: GPUComputePipeline;
  private fftColsPipeline!: GPUComputePipeline;
  private packPipeline!: GPUComputePipeline;
  private computeBindGroup!: GPUBindGroup;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private globalsBuffer: GPUBuffer,
    private sampleCount: number = 4,
  ) {}

  async init(): Promise<void> {
    this.uploadSpectrumBuffers();
    this.createTexture();
    await Promise.all([
      this.createComputePipelines(),
      this.createRenderPipeline(),
    ]);
  }

  private uploadSpectrumBuffers(): void {
    const rng    = mulberry32(42);
    const sz     = N * N;
    const h0Data   = new Float32Array(sz * 4);
    const freqData = new Float32Array(sz * 4);

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        const idx  = r * N + c;
        const nr   = r < N / 2 ? r : r - N;
        const nc   = c < N / 2 ? c : c - N;
        const kvx  = (2 * Math.PI * nc) / TILE_WORLD;
        const kvz  = (2 * Math.PI * nr) / TILE_WORLD;
        const kMag = Math.sqrt(kvx * kvx + kvz * kvz);

        freqData[idx * 4 + 0] = Math.sqrt(GRAVITY * Math.max(kMag, 1e-4));
        freqData[idx * 4 + 1] = kvx;
        freqData[idx * 4 + 2] = kvz;
        freqData[idx * 4 + 3] = 0;

        const sqrtPk  = Math.sqrt(phillipsSpectrum(kvx, kvz));
        const [g1r, g1i] = gaussRand(rng);
        h0Data[idx * 4 + 0] = (1 / Math.SQRT2) * g1r * sqrtPk;
        h0Data[idx * 4 + 1] = (1 / Math.SQRT2) * g1i * sqrtPk;

        const sqrtPnk = Math.sqrt(phillipsSpectrum(-kvx, -kvz));
        const [g2r, g2i] = gaussRand(rng);
        h0Data[idx * 4 + 2] =  (1 / Math.SQRT2) * g2r * sqrtPnk;
        h0Data[idx * 4 + 3] = -(1 / Math.SQRT2) * g2i * sqrtPnk;
      }
    }

    this.h0Buf = this.device.createBuffer({
      label: 'Ocean H0',
      size: h0Data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.h0Buf, 0, h0Data);

    this.freqBuf = this.device.createBuffer({
      label: 'Ocean Freq',
      size: freqData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.freqBuf, 0, freqData);

    this.specBuf = this.device.createBuffer({
      label: 'Ocean Spec',
      size: sz * 32,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  private createTexture(): void {
    this.oceanTex = this.device.createTexture({
      label: 'Ocean Data Texture',
      size: { width: N, height: N },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });
    this.oceanTexView = this.oceanTex.createView();
  }

  private async createComputePipelines(): Promise<void> {
    const module = this.device.createShaderModule({
      label: 'Ocean FFT',
      code: oceanFftShaderSource,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'Ocean Compute BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: 'rgba32float', viewDimension: '2d' },
        },
      ],
    });

    const layout = this.device.createPipelineLayout({
      label: 'Ocean Compute Layout',
      bindGroupLayouts: [bgl],
    });

    [this.spectrumPipeline, this.fftRowsPipeline, this.fftColsPipeline, this.packPipeline] =
      await Promise.all([
        this.device.createComputePipelineAsync({
          label: 'Ocean Spectrum', layout,
          compute: { module, entryPoint: 'spectrum_update' },
        }),
        this.device.createComputePipelineAsync({
          label: 'Ocean FFT Rows', layout,
          compute: { module, entryPoint: 'fft_rows' },
        }),
        this.device.createComputePipelineAsync({
          label: 'Ocean FFT Cols', layout,
          compute: { module, entryPoint: 'fft_cols' },
        }),
        this.device.createComputePipelineAsync({
          label: 'Ocean Pack', layout,
          compute: { module, entryPoint: 'ocean_pack' },
        }),
      ]);

    this.computeBindGroup = this.device.createBindGroup({
      label: 'Ocean Compute BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
        { binding: 1, resource: { buffer: this.h0Buf } },
        { binding: 2, resource: { buffer: this.freqBuf } },
        { binding: 3, resource: { buffer: this.specBuf } },
        { binding: 4, resource: this.oceanTexView },
      ],
    });
  }

  private async createRenderPipeline(): Promise<void> {
    const module = this.device.createShaderModule({
      label: 'Ocean Render',
      code: oceanShaderSource,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'Ocean Render BGL',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
        },
      ],
    });

    this.renderPipeline = await this.device.createRenderPipelineAsync({
      label: 'Ocean Render Pipeline',
      layout: this.device.createPipelineLayout({
        label: 'Ocean Render Layout',
        bindGroupLayouts: [bgl],
      }),
      vertex:   { module, entryPoint: 'vs_main' },
      fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
      primitive:    { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample:  { count: this.sampleCount },
    });

    this.renderBindGroup = this.device.createBindGroup({
      label: 'Ocean Render BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
        { binding: 1, resource: this.oceanTexView },
      ],
    });
  }

  encodeCompute(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: 'Ocean FFT' });
    pass.setBindGroup(0, this.computeBindGroup);

    pass.setPipeline(this.spectrumPipeline);
    pass.dispatchWorkgroups(N / 8, N / 8);

    pass.setPipeline(this.fftRowsPipeline);
    pass.dispatchWorkgroups(N);

    pass.setPipeline(this.fftColsPipeline);
    pass.dispatchWorkgroups(N);

    pass.setPipeline(this.packPipeline);
    pass.dispatchWorkgroups(N / 8, N / 8);

    pass.end();
  }

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.draw(MESH_N * MESH_N * 6);
  }
}
