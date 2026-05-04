import vegetationShaderSource from '../shaders/vegetation.wgsl?raw';
import grassShaderSource from '../shaders/grass.wgsl?raw';

const MAX_INSTANCES = 50000;
const SIZE = 512;

export class Vegetation {
  private drawArgsBuffer!: GPUBuffer;
  private positionsBuffer!: GPUBuffer;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private computePipeline!: GPUComputePipeline;
  private computeBG!: GPUBindGroup;
  private renderPipeline!: GPURenderPipeline;
  private renderBG!: GPUBindGroup;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private globalsBuffer: GPUBuffer,
    private heightmapTex: GPUTexture,
    private normalTex: GPUTexture,
    private smoothAccumTex: GPUTexture,
    private sampleCount: number = 4,
  ) {}

  async init(): Promise<void> {
    this.createBuffers();
    this.createMesh();
    await this.createComputePipeline();
    await this.createRenderPipeline();
  }

  private createBuffers(): void {
    this.drawArgsBuffer = this.device.createBuffer({
      label: 'Grass Draw Args',
      size: 20,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    });

    this.positionsBuffer = this.device.createBuffer({
      label: 'Grass Positions',
      size: MAX_INSTANCES * 16,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  private createMesh(): void {
    const verts = new Float32Array([
      -0.5, 0.0, 0.0, 0.0, 0.0,
       0.5, 0.0, 0.0, 1.0, 0.0,
      -0.5, 1.0, 0.0, 0.0, 1.0,
       0.5, 1.0, 0.0, 1.0, 1.0,
       0.0, 0.0, -0.5, 0.0, 0.0,
       0.0, 0.0, 0.5, 1.0, 0.0,
       0.0, 1.0, -0.5, 0.0, 1.0,
       0.0, 1.0, 0.5, 1.0, 1.0,
    ]);
    this.vertexBuffer = this.device.createBuffer({
      label: 'Foliage Vertex Buffer',
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, verts);

    const indices = new Uint16Array([
      0, 1, 2, 1, 3, 2,
      4, 5, 6, 5, 7, 6,
    ]);
    this.indexBuffer = this.device.createBuffer({
      label: 'Foliage Index Buffer',
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indexBuffer, 0, indices);
  }

  private async createComputePipeline(): Promise<void> {
    const bgl = this.device.createBindGroupLayout({
      label: 'Vegetation Compute BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
      ],
    });

    this.computePipeline = await this.device.createComputePipelineAsync({
      label: 'Vegetation Compute Pipeline',
      layout: this.device.createPipelineLayout({
        label: 'Veg Compute Layout',
        bindGroupLayouts: [bgl],
      }),
      compute: {
        module: this.device.createShaderModule({ label: 'Vegetation Shader', code: vegetationShaderSource }),
        entryPoint: 'place_vegetation',
      },
    });

    this.computeBG = this.device.createBindGroup({
      label: 'Vegetation Compute BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: this.heightmapTex.createView() },
        { binding: 1, resource: this.normalTex.createView() },
        { binding: 2, resource: { buffer: this.drawArgsBuffer } },
        { binding: 3, resource: { buffer: this.positionsBuffer } },
        { binding: 4, resource: this.smoothAccumTex.createView() },
      ],
    });
  }

  private async createRenderPipeline(): Promise<void> {
    const module = this.device.createShaderModule({ label: 'Grass Shader', code: grassShaderSource });

    const bgl = this.device.createBindGroupLayout({
      label: 'Grass Render BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    this.renderPipeline = await this.device.createRenderPipelineAsync({
      label: 'Grass Render Pipeline',
      layout: this.device.createPipelineLayout({ label: 'Grass Layout', bindGroupLayouts: [bgl] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 20,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x3' as GPUVertexFormat },
            { shaderLocation: 1, offset: 12, format: 'float32x2' as GPUVertexFormat },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: this.sampleCount },
    });

    this.renderBG = this.device.createBindGroup({
      label: 'Grass Render BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
        { binding: 1, resource: { buffer: this.positionsBuffer } },
      ],
    });
  }

  dispatchCompute(): void {
    this.device.queue.writeBuffer(this.drawArgsBuffer, 0, new Uint32Array([12, 0, 0, 0, 0]));

    const encoder = this.device.createCommandEncoder({ label: 'Vegetation Compute Encoder' });
    const pass = encoder.beginComputePass({ label: 'Place Vegetation' });
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.computeBG);
    pass.dispatchWorkgroups(SIZE / 8, SIZE / 8);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, this.renderBG);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint16');
    pass.drawIndexedIndirect(this.drawArgsBuffer, 0);
  }
}
