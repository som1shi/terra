import atmosphereShaderSource from '../shaders/atmosphere.wgsl?raw';
import precomputeShaderSource from '../shaders/atmosphere_precompute.wgsl?raw';

export class Atmosphere {
  private transmittanceTex!: GPUTexture;
  private pipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private indexCount!: number;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private globalsBuffer: GPUBuffer,
  ) {}

  async init(): Promise<void> {
    await this.precomputeTransmittance();
    await this.createRenderPipeline();
    this.createSkyQuad();
  }

  private async precomputeTransmittance(): Promise<void> {
    this.transmittanceTex = this.device.createTexture({
      label: 'Transmittance LUT',
      size: { width: 256, height: 64 },
      format: 'rgba32float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    const module = this.device.createShaderModule({
      label: 'Transmittance Precompute',
      code: precomputeShaderSource,
    });

    const pipeline = this.device.createComputePipeline({
      label: 'Transmittance Precompute Pipeline',
      layout: 'auto',
      compute: { module, entryPoint: 'compute_transmittance' },
    });

    const bg = this.device.createBindGroup({
      label: 'Transmittance Precompute BG',
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.transmittanceTex.createView() },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: 'Transmittance Precompute' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(256 / 8), Math.ceil(64 / 8));
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  private async createRenderPipeline(): Promise<void> {
    const module = this.device.createShaderModule({
      label: 'Atmosphere Render',
      code: atmosphereShaderSource,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'Atmosphere BGL',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float', viewDimension: '2d' },
        },
      ],
    });

    this.pipeline = this.device.createRenderPipeline({
      label: 'Atmosphere Pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 12,
          attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'zero' },
            alpha: { srcFactor: 'one', dstFactor: 'zero' },
          },
        }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample: { count: 4 },
    });

    this.bindGroup = this.device.createBindGroup({
      label: 'Atmosphere BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
        { binding: 1, resource: this.transmittanceTex.createView() },
      ],
    });
  }

  private createSkyQuad(): void {
    const verts = new Float32Array([
      -1, -1, 0.999,
       1, -1, 0.999,
       1,  1, 0.999,
      -1,  1, 0.999,
    ]);
    const idx = new Uint32Array([0, 1, 2, 0, 2, 3]);

    this.vertexBuffer = this.device.createBuffer({
      label: 'Sky Quad Verts',
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(verts);
    this.vertexBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
      label: 'Sky Quad Idx',
      size: idx.byteLength,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(idx);
    this.indexBuffer.unmap();

    this.indexCount = idx.length;
  }

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  destroy(): void {
    this.transmittanceTex?.destroy();
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
  }
}
