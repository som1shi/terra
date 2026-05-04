import { Camera } from './camera';
import { UI } from './ui';
import { Terrain } from './terrain/terrain';
import { Heightmap } from './terrain/heightmap';
import { ErosionSystem } from './terrain/erosion';
import { Vegetation } from './terrain/vegetation';
import { Ocean, SEA_LEVEL } from './terrain/ocean';
import { Atmosphere } from './atmosphere/atmosphere';
import { Bloom } from './post/bloom';

export const GLOBALS_BUFFER_SIZE = 208;

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';

export class Renderer {
  private camera: Camera;
  private globalsBuffer!: GPUBuffer;
  private depthTexture!: GPUTexture;
  private depthView!: GPUTextureView;

  private terrain!: Terrain;
  private ocean!: Ocean;
  private vegetation!: Vegetation;
  private atmosphere!: Atmosphere;
  private bloom!: Bloom;

  private msaaTexture!: GPUTexture;
  private msaaView!: GPUTextureView;
  private hdrTexture!: GPUTexture;
  private hdrView!: GPUTextureView;

  private timeOfDay = 0.45;
  private seed!: number;

  constructor(
    private device: GPUDevice,
    private context: GPUCanvasContext,
    private format: GPUTextureFormat,
    private canvas: HTMLCanvasElement,
    private ui: UI,
  ) {
    this.camera = new Camera(canvas);
  }

  async init(): Promise<void> {
    this.ui.setStatus('Allocating GPU resources...', 15);

    this.globalsBuffer = this.device.createBuffer({
      label: 'Globals Uniform Buffer',
      size: GLOBALS_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.createDepthTexture();
    this.createHDRTextures();

    this.ui.setStatus('Generating terrain heightmap...', 40);

    const urlSeed = new URLSearchParams(window.location.search).get('seed');
    this.seed = urlSeed ? (parseInt(urlSeed, 10) || 137) : 137;

    const heightmap = new Heightmap(512, 512, this.seed);
    const heightData = heightmap.generate();

    const heightmapTex = this.device.createTexture({
      label: 'Heightmap Texture',
      size: { width: 512, height: 512 },
      format: 'r32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING,
    });

    this.device.queue.writeTexture(
      { texture: heightmapTex },
      heightData,
      { bytesPerRow: 512 * 4 },
      { width: 512, height: 512 },
    );

    this.ui.setStatus('Running hydraulic erosion...', 60);

    const erosion = new ErosionSystem(this.device);
    const erosionEncoder = this.device.createCommandEncoder({ label: 'Erosion Encoder' });
    erosion.run(erosionEncoder, heightmapTex);
    this.device.queue.submit([erosionEncoder.finish()]);

    this.ui.setStatus('Building terrain mesh...', 75);

    this.terrain = new Terrain(this.device, HDR_FORMAT, heightmapTex, this.globalsBuffer, erosion.getSmoothedAccumTex());
    await this.terrain.init();

    this.ocean = new Ocean(this.device, HDR_FORMAT, this.globalsBuffer, 4);
    await this.ocean.init();

    this.ui.setStatus('Placing vegetation...', 90);

    this.vegetation = new Vegetation(
      this.device,
      HDR_FORMAT,
      this.globalsBuffer,
      heightmapTex,
      this.terrain.getNormalTex(),
      erosion.getSmoothedAccumTex(),
      4,
    );
    await this.vegetation.init();
    this.vegetation.dispatchCompute();

    this.ui.setStatus('Initializing atmosphere...', 95);

    try {
      this.atmosphere = new Atmosphere(this.device, HDR_FORMAT, this.globalsBuffer, this.seed);
      await this.atmosphere.init();
    } catch (error) {
      console.error('Renderer: Failed to initialize atmosphere:', error);
      throw error;
    }

    this.bloom = new Bloom(this.device, this.format);
    await this.bloom.init(this.canvas.width, this.canvas.height);

    this.ui.setStatus('Ready!', 100);
  }

  private createDepthTexture(): void {
    if (this.depthTexture) this.depthTexture.destroy();
    this.depthTexture = this.device.createTexture({
      label: 'Depth Texture',
      size: { width: this.canvas.width, height: this.canvas.height },
      format: 'depth24plus',
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depthTexture.createView();
  }

  private createHDRTextures(): void {
    if (this.msaaTexture) this.msaaTexture.destroy();
    if (this.hdrTexture) this.hdrTexture.destroy();
    this.msaaTexture = this.device.createTexture({
      label: 'MSAA HDR Texture',
      size: { width: this.canvas.width, height: this.canvas.height },
      format: HDR_FORMAT,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.msaaView = this.msaaTexture.createView();

    this.hdrTexture = this.device.createTexture({
      label: 'HDR Resolve Texture',
      size: { width: this.canvas.width, height: this.canvas.height },
      format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.hdrView = this.hdrTexture.createView();
  }

  onResize(width: number, height: number): void {
    this.camera.onResize(width, height);
    this.createDepthTexture();
    this.createHDRTextures();
    this.bloom?.resize(width, height);
  }

  setTimeOfDay(tod: number): void {
    this.timeOfDay = tod;
  }

  getCameraPosition(): [number, number, number] {
    return this.camera.position;
  }

  render(time: number, dt: number): void {
    this.camera.update(dt);

    const theta = this.timeOfDay * Math.PI * 2 - Math.PI * 0.5;
    const sunY = Math.sin(theta);
    const sunX = Math.cos(theta) * 0.5;
    const sunZ = -Math.cos(theta) * 0.866;
    const sunLen = Math.sqrt(sunX * sunX + sunY * sunY + sunZ * sunZ);

    // Moon is opposite the sun (full-moon arc: at zenith at midnight, below horizon at noon)
    const moonDirX = -sunX / sunLen;
    const moonDirY = -sunY / sunLen;
    const moonDirZ = -sunZ / sunLen;

    // moonIntensity: 0 when below horizon or during day, 1 when moon at zenith at midnight
    const sunElevation = sunY / sunLen;
    const dayFactor = Math.max(0, Math.min(1, (sunElevation + 0.05) / 0.20));
    const nightFactor = 1.0 - dayFactor;
    const moonIntensity = Math.max(0, moonDirY) * nightFactor;

    const globalsData = new ArrayBuffer(GLOBALS_BUFFER_SIZE);
    const f32 = new Float32Array(globalsData);
    const view = new DataView(globalsData);

    const vp = this.camera.viewProjMatrix;
    for (let i = 0; i < 16; i++) f32[i] = vp[i];

    const invVP = this.camera.getInverseViewProj();
    for (let i = 0; i < 16; i++) f32[16 + i] = invVP[i];

    f32[32] = sunX / sunLen;
    f32[33] = sunY / sunLen;
    f32[34] = sunZ / sunLen;
    f32[35] = 0;

    const cp = this.camera.position;
    f32[36] = cp[0];
    f32[37] = cp[1];
    f32[38] = cp[2];
    f32[39] = 0;

    view.setFloat32(160, time, true);
    view.setFloat32(164, this.timeOfDay, true);
    view.setFloat32(168, SEA_LEVEL, true);
    view.setFloat32(172, this.seed, true);
    view.setFloat32(176, this.canvas.width, true);
    view.setFloat32(180, this.canvas.height, true);
    view.setFloat32(184, moonIntensity, true);  // offset 184: moonIntensity (was _pad3.x)
    // offset 188: _pad2 (zero-initialized)
    view.setFloat32(192, moonDirX, true);       // offset 192: moonDir.x
    view.setFloat32(196, moonDirY, true);       // offset 196: moonDir.y
    view.setFloat32(200, moonDirZ, true);       // offset 200: moonDir.z
    // offset 204: _pad3 (zero-initialized)

    this.device.queue.writeBuffer(this.globalsBuffer, 0, globalsData);

    const encoder = this.device.createCommandEncoder({ label: 'Frame Encoder' });

    this.ocean.encodeCompute(encoder);

    const renderPass = encoder.beginRenderPass({
      label: 'Main Render Pass',
      colorAttachments: [{
        view: this.msaaView,
        resolveTarget: this.hdrView,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'discard',
      }],
      depthStencilAttachment: {
        view: this.depthView,
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    });

    this.atmosphere.encode(renderPass);
    this.terrain.encode(renderPass);
    this.ocean.encode(renderPass);
    this.vegetation.encode(renderPass);

    renderPass.end();
    this.bloom.encode(encoder, this.hdrTexture, this.context.getCurrentTexture().createView());
    this.device.queue.submit([encoder.finish()]);
  }
}
