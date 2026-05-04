import bloomComputeSource from '../shaders/bloom_compute.wgsl?raw';
import bloomCompositeSource from '../shaders/bloom_composite.wgsl?raw';

const BLOOM_PASSES = 4;

export class Bloom {
  private thresholdPipeline!: GPUComputePipeline;
  private downsamplePipeline!: GPUComputePipeline;
  private upsamplePipeline!: GPUComputePipeline;
  private compositePipeline!: GPURenderPipeline;
  private linearSampler!: GPUSampler;

  private pingTex!: GPUTexture;
  private pongTex!: GPUTexture;
  private offsetBuf!: GPUBuffer;

  private thresholdBGL!: GPUBindGroupLayout;
  private blurBGL!: GPUBindGroupLayout;
  private compositeBGL!: GPUBindGroupLayout;

  private width = 0;
  private height = 0;

  constructor(private device: GPUDevice, private canvasFormat: GPUTextureFormat) {}

  async init(width: number, height: number): Promise<void> {
    this.width = width;
    this.height = height;

    this.linearSampler = this.device.createSampler({
      label: 'Bloom Sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.offsetBuf = this.device.createBuffer({
      label: 'Blur Offset',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.createBloomTextures(width, height);
    await this.createPipelines();
  }

  private createBloomTextures(w: number, h: number): void {
    const hw = Math.max(1, Math.ceil(w / 2));
    const hh = Math.max(1, Math.ceil(h / 2));
    this.pingTex?.destroy();
    this.pongTex?.destroy();
    const desc: GPUTextureDescriptor = {
      size: { width: hw, height: hh },
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    };
    this.pingTex = this.device.createTexture({ ...desc, label: 'Bloom Ping' });
    this.pongTex = this.device.createTexture({ ...desc, label: 'Bloom Pong' });
  }

  private async createPipelines(): Promise<void> {
    const computeModule = this.device.createShaderModule({ label: 'Bloom Compute', code: bloomComputeSource });
    const compositeModule = this.device.createShaderModule({ label: 'Bloom Composite', code: bloomCompositeSource });

    this.thresholdBGL = this.device.createBindGroupLayout({
      label: 'Threshold BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    this.blurBGL = this.device.createBindGroupLayout({
      label: 'Blur BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    this.compositeBGL = this.device.createBindGroupLayout({
      label: 'Composite BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });

    const emptyBGL = this.device.createBindGroupLayout({ label: 'Empty BGL', entries: [] });
    const blurLayout = this.device.createPipelineLayout({
      label: 'Blur Layout',
      bindGroupLayouts: [emptyBGL, this.blurBGL],
    });

    [this.thresholdPipeline, this.downsamplePipeline, this.upsamplePipeline] = await Promise.all([
      this.device.createComputePipelineAsync({
        label: 'Bloom Threshold',
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.thresholdBGL] }),
        compute: { module: computeModule, entryPoint: 'bloom_threshold' },
      }),
      this.device.createComputePipelineAsync({
        label: 'Bloom Downsample',
        layout: blurLayout,
        compute: { module: computeModule, entryPoint: 'blur_downsample' },
      }),
      this.device.createComputePipelineAsync({
        label: 'Bloom Upsample',
        layout: blurLayout,
        compute: { module: computeModule, entryPoint: 'blur_upsample' },
      }),
    ]);

    this.compositePipeline = await this.device.createRenderPipelineAsync({
      label: 'Bloom Composite',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.compositeBGL] }),
      vertex: { module: compositeModule, entryPoint: 'vs_main' },
      fragment: { module: compositeModule, entryPoint: 'fs_main', targets: [{ format: this.canvasFormat }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.createBloomTextures(width, height);
  }

  encode(encoder: GPUCommandEncoder, hdrTex: GPUTexture, canvasView: GPUTextureView): void {
    const hw = Math.max(1, Math.ceil(this.width / 2));
    const hh = Math.max(1, Math.ceil(this.height / 2));

    const thrBG = this.device.createBindGroup({
      layout: this.thresholdBGL,
      entries: [
        { binding: 0, resource: hdrTex.createView() },
        { binding: 1, resource: this.pingTex.createView() },
      ],
    });
    {
      const p = encoder.beginComputePass({ label: 'Bloom Threshold' });
      p.setPipeline(this.thresholdPipeline);
      p.setBindGroup(0, thrBG);
      p.dispatchWorkgroups(Math.ceil(hw / 8), Math.ceil(hh / 8));
      p.end();
    }

    const emptyBG = this.device.createBindGroup({
      layout: this.downsamplePipeline.getBindGroupLayout(0),
      entries: [],
    });

    let src = this.pingTex;
    let dst = this.pongTex;

    for (let i = 0; i < BLOOM_PASSES; i++) {
      this.device.queue.writeBuffer(this.offsetBuf, 0, new Float32Array([i + 1, 0, 0, 0]));

      const blurBG = this.device.createBindGroup({
        layout: this.blurBGL,
        entries: [
          { binding: 0, resource: src.createView() },
          { binding: 1, resource: dst.createView() },
          { binding: 2, resource: this.linearSampler },
          { binding: 3, resource: { buffer: this.offsetBuf } },
        ],
      });

      const half = Math.floor(BLOOM_PASSES / 2);
      const pipeline = i < half ? this.downsamplePipeline : this.upsamplePipeline;

      const p = encoder.beginComputePass({ label: `Bloom Blur ${i}` });
      p.setPipeline(pipeline);
      p.setBindGroup(0, emptyBG);
      p.setBindGroup(1, blurBG);
      p.dispatchWorkgroups(Math.ceil(hw / 8), Math.ceil(hh / 8));
      p.end();

      [src, dst] = [dst, src];
    }

    const bloomResult = src;

    const compBG = this.device.createBindGroup({
      layout: this.compositeBGL,
      entries: [
        { binding: 0, resource: hdrTex.createView() },
        { binding: 1, resource: bloomResult.createView() },
        { binding: 2, resource: this.linearSampler },
      ],
    });

    const pass = encoder.beginRenderPass({
      label: 'Bloom Composite',
      colorAttachments: [{
        view: canvasView,
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
      }],
    });
    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, compBG);
    pass.draw(3);
    pass.end();
  }

  destroy(): void {
    this.pingTex?.destroy();
    this.pongTex?.destroy();
    this.offsetBuf?.destroy();
  }
}
