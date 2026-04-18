import terrainShaderSource from '../shaders/terrain.wgsl?raw';
import normalsShaderSource from '../shaders/normals.wgsl?raw';
import aoShaderSource from '../shaders/ao.wgsl?raw';

export class Terrain {
  private pipeline!: GPURenderPipeline;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private indexCount!: number;
  private bindGroup!: GPUBindGroup;
  private heightmapSampler!: GPUSampler;
  private normalTex!: GPUTexture;
  private normalSampler!: GPUSampler;
  private aoTex!: GPUTexture;
  private aoSampler!: GPUSampler;

  private readonly GRID = 512;
  private readonly WORLD_SCALE = 4096;
  private readonly HEIGHT_SCALE = 600;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private heightmapTex: GPUTexture,
    private globalsBuffer: GPUBuffer,
    private accumTex: GPUTexture,
  ) { }

  async init(): Promise<void> {
    this.heightmapSampler = this.device.createSampler({
      label: 'Heightmap Sampler',
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.normalSampler = this.device.createSampler({
      label: 'Normal Map Sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.aoSampler = this.device.createSampler({
      label: 'AO Sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.computeNormalMap();
    this.computeAO();
    this.buildMesh();
    await this.createPipeline();
    this.createBindGroup();
  }

  private computeNormalMap(): void {
    this.normalTex = this.device.createTexture({
      label: 'Normal Map',
      size: { width: 512, height: 512 },
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'Normals BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: 'Normals Pipeline',
      layout: this.device.createPipelineLayout({ label: 'Normals Layout', bindGroupLayouts: [bgl] }),
      compute: {
        module: this.device.createShaderModule({ label: 'Normals Shader', code: normalsShaderSource }),
        entryPoint: 'build_normals',
      },
    });

    const bg = this.device.createBindGroup({
      label: 'Normals BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: this.heightmapTex.createView() },
        { binding: 1, resource: this.normalTex.createView() },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: 'Normals Encoder' });
    const pass = encoder.beginComputePass({ label: 'Normals Pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(64, 64);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private computeAO(): void {
    this.aoTex = this.device.createTexture({
      label: 'HBAO Texture',
      size: { width: 512, height: 512 },
      format: 'rgba16float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'AO BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });

    const pipeline = this.device.createComputePipeline({
      label: 'AO Pipeline',
      layout: this.device.createPipelineLayout({ label: 'AO Layout', bindGroupLayouts: [bgl] }),
      compute: {
        module: this.device.createShaderModule({ label: 'AO Shader', code: aoShaderSource }),
        entryPoint: 'build_ao',
      },
    });

    const bg = this.device.createBindGroup({
      label: 'AO BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: this.heightmapTex.createView() },
        { binding: 1, resource: this.aoTex.createView() },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: 'AO Encoder' });
    const pass = encoder.beginComputePass({ label: 'AO Pass' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(64, 64);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  private buildMesh(): void {
    const N = this.GRID;
    const half = this.WORLD_SCALE / 2;

    const vertData = new Float32Array(N * N * 4);
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < N; i++) {
        const idx = (j * N + i) * 4;
        vertData[idx + 0] = (i / (N - 1)) * this.WORLD_SCALE - half;
        vertData[idx + 1] = (j / (N - 1)) * this.WORLD_SCALE - half;
        vertData[idx + 2] = i / (N - 1);
        vertData[idx + 3] = j / (N - 1);
      }
    }

    this.vertexBuffer = this.device.createBuffer({
      label: 'Terrain Vertex Buffer',
      size: vertData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertData);

    const quadCount = (N - 1) * (N - 1);
    const indices = new Uint32Array(quadCount * 6);
    let idx = 0;
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const tl = j * N + i;
        const tr = tl + 1;
        const bl = tl + N;
        const br = bl + 1;
        indices[idx++] = tl; indices[idx++] = bl; indices[idx++] = br;
        indices[idx++] = tl; indices[idx++] = br; indices[idx++] = tr;
      }
    }
    this.indexCount = idx;

    this.indexBuffer = this.device.createBuffer({
      label: 'Terrain Index Buffer',
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indexBuffer, 0, indices);
  }

  private async createPipeline(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: 'Terrain Shader',
      code: terrainShaderSource,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'Terrain BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'unfilterable-float' } },
      ],
    });

    this.pipeline = await this.device.createRenderPipelineAsync({
      label: 'Terrain Pipeline',
      layout: this.device.createPipelineLayout({
        label: 'Terrain Pipeline Layout',
        bindGroupLayouts: [bgl],
      }),
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 16,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 8, format: 'float32x2' },
          ],
        }],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: 4 },
    });
  }

  private createBindGroup(): void {
    this.bindGroup = this.device.createBindGroup({
      label: 'Terrain Bind Group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
        { binding: 1, resource: this.heightmapTex.createView() },
        { binding: 2, resource: this.heightmapSampler },
        { binding: 3, resource: this.normalTex.createView() },
        { binding: 4, resource: this.normalSampler },
        { binding: 5, resource: this.aoTex.createView() },
        { binding: 6, resource: this.aoSampler },
        { binding: 7, resource: this.accumTex.createView() },
      ],
    });
  }

  getNormalTex(): GPUTexture { return this.normalTex; }

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  getWorldScale(): number { return this.WORLD_SCALE; }
  getHeightScale(): number { return this.HEIGHT_SCALE; }
}
