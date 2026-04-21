import atmosphereShaderSource from '../shaders/atmosphere.wgsl?raw';

export type ClimatePreset = 'temperate' | 'arid' | 'tropical' | 'arctic' | 'stormy';
export const CLIMATE_PRESETS: ClimatePreset[] = ['temperate','arid','tropical','arctic','stormy'];

const PRESET_INDEX: Record<ClimatePreset, number> = {
  temperate: 0, arid: 1, tropical: 2, arctic: 3, stormy: 4,
};

export class Atmosphere {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private globalsBuffer: GPUBuffer;

  private transmittanceTexture!: GPUTexture;
  private scatteringTexture!: GPUTexture;
  private irradianceTexture!: GPUTexture;

  private pipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;
  private climateBG!: GPUBindGroup;
  private climateBuffer!: GPUBuffer;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private indexCount!: number;

  private currentPreset: ClimatePreset = 'temperate';
  private seed: number;

  constructor(device: GPUDevice, format: GPUTextureFormat, globalsBuffer: GPUBuffer, seed: number) {
    this.device        = device;
    this.format        = format;
    this.globalsBuffer = globalsBuffer;
    this.seed          = seed;
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  setPreset(preset: ClimatePreset): void {
    this.currentPreset = preset;
    this.uploadClimateBuffer();
  }

  getPreset(): ClimatePreset { return this.currentPreset; }

  // ── Init ─────────────────────────────────────────────────────────────────────

  async init(): Promise<void> {
    this.createClimateBuffer();
    await this.createPrecomputedTextures();
    await this.createRenderPipeline();
    this.createSkyDome();
  }

  private createClimateBuffer(): void {
    this.climateBuffer = this.device.createBuffer({
      label: 'Climate Uniform', size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.uploadClimateBuffer();
  }

  private uploadClimateBuffer(): void {
    const data = new Float32Array(4);
    data[0] = PRESET_INDEX[this.currentPreset];
    this.device.queue.writeBuffer(this.climateBuffer, 0, data);
  }

  // ── Precompute ───────────────────────────────────────────────────────────────

  private async createPrecomputedTextures(): Promise<void> {
    this.transmittanceTexture = this.device.createTexture({
      label: 'Atmosphere Transmittance', size: { width: 256, height: 64 },
      format: 'rg16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.scatteringTexture = this.device.createTexture({
      label: 'Atmosphere Scattering', size: { width: 256, height: 128, depthOrArrayLayers: 32 },
      format: 'rgba16float', dimension: '3d',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.irradianceTexture = this.device.createTexture({
      label: 'Atmosphere Irradiance', size: { width: 64, height: 16 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    await this.precomputeAtmosphere();
  }

  private async precomputeAtmosphere(): Promise<void> {
    const computeShader = this.device.createShaderModule({
      label: 'Atmosphere Precompute',
      code: `
        @group(0) @binding(0) var transmittance_tex : texture_storage_2d<rg16float, write>;
        @group(0) @binding(1) var scattering_tex    : texture_storage_3d<rgba16float, write>;
        @group(0) @binding(2) var irradiance_tex    : texture_storage_2d<rgba16float, write>;
        const EARTH_RADIUS = 6371000.0; const ATMOSPHERE_HEIGHT = 100000.0;
        fn atmos_h(r: f32) -> f32 { return sqrt(max(r*r - EARTH_RADIUS*EARTH_RADIUS, 0.0)); }
        fn transmittance(r: f32, mu: f32) -> vec2<f32> {
          let atop = EARTH_RADIUS + ATMOSPHERE_HEIGHT;
          let d = max(0.0, sqrt(max(r*r*(mu*mu-1.0)+atop*atop, 0.0)) - r*mu);
          return vec2<f32>(exp(-exp(-atmos_h(r)/8000.0)*d), exp(-exp(-atmos_h(r)/1200.0)*d));
        }
        @compute @workgroup_size(8,8)
        fn compute_transmittance_main(@builtin(global_invocation_id) id: vec3<u32>) {
          let sz = textureDimensions(transmittance_tex);
          if (id.x >= sz.x || id.y >= sz.y) { return; }
          let r = EARTH_RADIUS + (f32(id.x)+0.5)/f32(sz.x)*ATMOSPHERE_HEIGHT;
          let mu = 2.0*(f32(id.y)+0.5)/f32(sz.y) - 1.0;
          textureStore(transmittance_tex, vec2<i32>(id.xy), vec4<f32>(transmittance(r,mu), 0.0, 1.0));
        }
      `,
    });
    const pipeline = this.device.createComputePipeline({
      label: 'Atmos Precompute', layout: 'auto',
      compute: { module: computeShader, entryPoint: 'compute_transmittance_main' },
    });
    const bg = this.device.createBindGroup({
      label: 'Atmos Precompute BG', layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.transmittanceTexture.createView() },
        { binding: 1, resource: this.scatteringTexture.createView()    },
        { binding: 2, resource: this.irradianceTexture.createView()    },
      ],
    });
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(256/8), Math.ceil(64/8));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  // ── Render pipeline ──────────────────────────────────────────────────────────

  private async createRenderPipeline(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: 'Atmosphere Shader', code: atmosphereShaderSource,
    });

    const bgl0 = this.device.createBindGroupLayout({
      label: 'Atmos BGL0',
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });
    const bgl1 = this.device.createBindGroupLayout({
      label: 'Atmos BGL1 Climate',
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } }],
    });

    this.pipeline = this.device.createRenderPipeline({
      label: 'Atmosphere Pipeline',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [bgl0, bgl1] }),
      vertex: {
        module: shaderModule, entryPoint: 'vs_main',
        buffers: [{ arrayStride: 12, attributes: [{ format: 'float32x3', offset: 0, shaderLocation: 0 }] }],
      },
      fragment: {
        module: shaderModule, entryPoint: 'fs_main',
        targets: [{ format: this.format, blend: {
          color: { srcFactor: 'one', dstFactor: 'zero' },
          alpha: { srcFactor: 'one', dstFactor: 'zero' },
        }}],
      },
      primitive:    { topology: 'triangle-list', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: false, depthCompare: 'less-equal' },
      multisample:  { count: 4 },
    });

    this.bindGroup = this.device.createBindGroup({
      label: 'Atmos BG0', layout: bgl0,
      entries: [{ binding: 0, resource: { buffer: this.globalsBuffer } }],
    });
    this.climateBG = this.device.createBindGroup({
      label: 'Atmos BG1 Climate', layout: bgl1,
      entries: [{ binding: 0, resource: { buffer: this.climateBuffer } }],
    });
  }

  // ── Sky quad ─────────────────────────────────────────────────────────────────

  private createSkyDome(): void {
    const verts   = new Float32Array([-1,-1,0.999, 1,-1,0.999, 1,1,0.999, -1,1,0.999]);
    const indices = new Uint32Array([0,1,2, 0,2,3]);
    this.vertexBuffer = this.device.createBuffer({
      label: 'Sky Quad Verts', size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX, mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(verts);
    this.vertexBuffer.unmap();
    this.indexBuffer = this.device.createBuffer({
      label: 'Sky Quad Indices', size: indices.byteLength,
      usage: GPUBufferUsage.INDEX, mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();
    this.indexCount = indices.length;
  }

  // ── Encode ───────────────────────────────────────────────────────────────────

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setBindGroup(1, this.climateBG);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  destroy(): void {
    this.transmittanceTexture?.destroy();
    this.scatteringTexture?.destroy();
    this.irradianceTexture?.destroy();
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
    this.climateBuffer?.destroy();
  }
}
