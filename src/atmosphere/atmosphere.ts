import atmosphereShaderSource from '../shaders/atmosphere.wgsl?raw';

export class Atmosphere {
  private transmittanceTexture!: GPUTexture;
  private scatteringTexture!: GPUTexture;
  private irradianceTexture!: GPUTexture;
  private pipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;
  private indexCount!: number;

  private readonly EARTH_RADIUS = 6371000.0;
  private readonly ATMOSPHERE_HEIGHT = 100000.0;
  private readonly RAYLEIGH_SCATTERING = [5.802e-6, 13.558e-6, 33.1e-6];
  private readonly MIE_SCATTERING = 3.996e-6;
  private readonly MIE_EXTINCTION = 4.44e-6;
  private readonly MIE_PHASE_G = 0.8;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private globalsBuffer: GPUBuffer,
    private seed: number,
  ) {}

  async init(): Promise<void> {
    await this.createPrecomputedTextures();
    await this.createRenderPipeline();
    this.createSkyDome();
  }

  private async createPrecomputedTextures(): Promise<void> {
    this.transmittanceTexture = this.device.createTexture({
      label: 'Atmosphere Transmittance',
      size: { width: 256, height: 64 },
      format: 'rg16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.scatteringTexture = this.device.createTexture({
      label: 'Atmosphere Scattering',
      size: { width: 256, height: 128, depthOrArrayLayers: 32 },
      format: 'rgba16float',
      dimension: '3d',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this.irradianceTexture = this.device.createTexture({
      label: 'Atmosphere Irradiance',
      size: { width: 64, height: 16 },
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    await this.precomputeAtmosphere();
  }

  private async precomputeAtmosphere(): Promise<void> {
    const computeShader = this.device.createShaderModule({
      label: 'Atmosphere Precompute Shader',
      code: `
        struct AtmosphereParams {
          earth_radius: f32,
          atmosphere_height: f32,
          rayleigh_scattering: vec3<f32>,
          mie_scattering: f32,
          mie_extinction: f32,
          mie_phase_g: f32,
        };

        @group(0) @binding(0) var transmittance_tex: texture_storage_2d<rg16float, write>;
        @group(0) @binding(1) var scattering_tex: texture_storage_3d<rgba16float, write>;
        @group(0) @binding(2) var irradiance_tex: texture_storage_2d<rgba16float, write>;

        const EARTH_RADIUS = 6371000.0;
        const ATMOSPHERE_HEIGHT = 100000.0;
        const RAYLEIGH_SCATTERING = vec3<f32>(5.802e-6, 13.558e-6, 33.1e-6);
        const MIE_SCATTERING = 3.996e-6;
        const MIE_EXTINCTION = 4.44e-6;
        const MIE_PHASE_G = 0.8;

        fn atmosphere_height_at_radius(r: f32) -> f32 {
          return sqrt(r * r - EARTH_RADIUS * EARTH_RADIUS);
        }

        fn optical_depth_rayleigh(r: f32, mu: f32, d: f32) -> f32 {
          let h = atmosphere_height_at_radius(r);
          let scale_height = 8000.0;
          return exp(-h / scale_height) * d;
        }

        fn optical_depth_mie(r: f32, mu: f32, d: f32) -> f32 {
          let h = atmosphere_height_at_radius(r);
          let scale_height = 1200.0;
          return exp(-h / scale_height) * d;
        }

        fn compute_transmittance(r: f32, mu: f32) -> vec2<f32> {
          let atmosphere_top = EARTH_RADIUS + ATMOSPHERE_HEIGHT;
          let discriminant = r * r * (mu * mu - 1.0) + atmosphere_top * atmosphere_top;
          let d = max(0.0, sqrt(discriminant) - r * mu);

          let rayleigh_optical_depth = optical_depth_rayleigh(r, mu, d);
          let mie_optical_depth = optical_depth_mie(r, mu, d);

          let rayleigh_transmittance = exp(-rayleigh_optical_depth);
          let mie_transmittance = exp(-mie_optical_depth);

          return vec2<f32>(rayleigh_transmittance, mie_transmittance);
        }

        @compute @workgroup_size(8, 8)
        fn compute_transmittance_main(@builtin(global_invocation_id) id: vec3<u32>) {
          let tex_size = textureDimensions(transmittance_tex);
          if (id.x >= tex_size.x || id.y >= tex_size.y) { return; }

          let u_r = (f32(id.x) + 0.5) / f32(tex_size.x);
          let u_mu = (f32(id.y) + 0.5) / f32(tex_size.y);

          let r = EARTH_RADIUS + u_r * ATMOSPHERE_HEIGHT;
          let mu = 2.0 * u_mu - 1.0;

          let transmittance = compute_transmittance(r, mu);
          textureStore(transmittance_tex, vec2<i32>(id.xy), vec4<f32>(transmittance, 0.0, 1.0));
        }
      `,
    });

    const computePipeline = this.device.createComputePipeline({
      label: 'Atmosphere Precompute Pipeline',
      layout: 'auto',
      compute: {
        module: computeShader,
        entryPoint: 'compute_transmittance_main',
      },
    });

    const bindGroup = this.device.createBindGroup({
      label: 'Atmosphere Precompute Bind Group',
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.transmittanceTexture.createView() },
        { binding: 1, resource: this.scatteringTexture.createView() },
        { binding: 2, resource: this.irradianceTexture.createView() },
      ],
    });

    const encoder = this.device.createCommandEncoder({ label: 'Atmosphere Precompute' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(computePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(256 / 8), Math.ceil(64 / 8), 1);
    pass.end();
    this.device.queue.submit([encoder.finish()]);

    await this.device.queue.onSubmittedWorkDone();
  }

  private async createRenderPipeline(): Promise<void> {
    const shaderModule = this.device.createShaderModule({
      label: 'Atmosphere Shader',
      code: atmosphereShaderSource,
    });

    this.pipeline = this.device.createRenderPipeline({
      label: 'Atmosphere Pipeline',
      layout: 'auto',
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 12,
          attributes: [{
            format: 'float32x3',
            offset: 0,
            shaderLocation: 0,
          }],
        }],
      },
      fragment: {
        module: shaderModule,
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
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: false,
        depthCompare: 'less-equal',
      },
      multisample: { count: 4 },
    });

    this.bindGroup = this.device.createBindGroup({
      label: 'Atmosphere Bind Group',
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer } },
      ],
    });
  }

  private createSkyDome(): void {
    const vertices = [
      -1.0, -1.0, 0.999,
       1.0, -1.0, 0.999,
       1.0,  1.0, 0.999,
      -1.0,  1.0, 0.999,
    ];

    const indices = [0, 1, 2, 0, 2, 3];

    this.vertexBuffer = this.device.createBuffer({
      label: 'Sky Dome Vertices',
      size: vertices.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
    this.vertexBuffer.unmap();

    this.indexCount = indices.length;

    this.indexBuffer = this.device.createBuffer({
      label: 'Sky Dome Indices',
      size: indices.length * 4,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();
  }

  encode(renderPass: GPURenderPassEncoder): void {
    renderPass.setPipeline(this.pipeline);
    renderPass.setBindGroup(0, this.bindGroup);
    renderPass.setVertexBuffer(0, this.vertexBuffer);
    renderPass.setIndexBuffer(this.indexBuffer, 'uint32');
    renderPass.drawIndexed(this.indexCount);
  }

  destroy(): void {
    this.transmittanceTexture?.destroy();
    this.scatteringTexture?.destroy();
    this.irradianceTexture?.destroy();
    this.vertexBuffer?.destroy();
    this.indexBuffer?.destroy();
  }
}
