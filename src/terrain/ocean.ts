import oceanShaderSource from '../shaders/ocean.wgsl?raw';

export const SEA_LEVEL = 5.0;

export class Ocean {
  private pipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private globalsBuffer: GPUBuffer,
    private sampleCount: number = 4,
  ) { }

  async init(): Promise<void> {
    await this.createPipeline();
  }

  private async createPipeline(): Promise<void> {
    const module = this.device.createShaderModule({
      label: 'Ocean Shader',
      code: oceanShaderSource,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'Ocean BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });

    this.pipeline = await this.device.createRenderPipelineAsync({
      label: 'Ocean Pipeline',
      layout: this.device.createPipelineLayout({ label: 'Ocean Layout', bindGroupLayouts: [bgl] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
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
      label: 'Ocean Bind Group',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
      ],
    });
  }

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
  }
}
