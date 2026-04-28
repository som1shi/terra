import { Camera } from './camera';
import { UI } from './ui';
import { Terrain } from './terrain/terrain';
import { Heightmap } from './terrain/heightmap';
import { ErosionSystem } from './terrain/erosion';
import { Vegetation } from './terrain/vegetation';
import { Ocean, SEA_LEVEL } from './terrain/ocean';
import { Atmosphere } from './atmosphere/atmosphere';
import { Flock } from './birds/flock';

export const GLOBALS_BUFFER_SIZE = 176;

export class Renderer {
  private camera: Camera;
  private globalsBuffer!: GPUBuffer;
  private depthTexture!: GPUTexture;
  private depthView!: GPUTextureView;

  private terrain!: Terrain;
  private ocean!: Ocean;
  private vegetation!: Vegetation;
  private atmosphere!: Atmosphere;
  private flock!: Flock;
  private msaaTexture!: GPUTexture;
  private msaaView!: GPUTextureView;
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
    this.createMSAATexture();

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

    this.terrain = new Terrain(this.device, this.format, heightmapTex, this.globalsBuffer, erosion.getSmoothedAccumTex());
    await this.terrain.init();

    this.ocean = new Ocean(this.device, this.format, this.globalsBuffer, 4);
    await this.ocean.init();

    this.ui.setStatus('Placing vegetation...', 90);

    this.vegetation = new Vegetation(
      this.device,
      this.format,
      this.globalsBuffer,
      heightmapTex,
      this.terrain.getNormalTex(),
      erosion.getSmoothedAccumTex(),
      4,
    );
    await this.vegetation.init();
    this.vegetation.dispatchCompute();

    this.ui.setStatus('Initializing atmosphere...', 93);

    try {
      this.atmosphere = new Atmosphere(this.device, this.format, this.globalsBuffer, this.seed);
      await this.atmosphere.init();
      console.log('Atmosphere: Successfully integrated into renderer');
    } catch (error) {
      console.error('Renderer: Failed to initialize atmosphere:', error);
      throw error;
    }

    this.ui.setStatus('Spawning birds...', 97);
    this.flock = new Flock(this.device, this.format, this.globalsBuffer, 4);
    await this.flock.initGPU();

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

  private createMSAATexture(): void {
    if (this.msaaTexture) this.msaaTexture.destroy();
    this.msaaTexture = this.device.createTexture({
      label: 'MSAA Color Texture',
      size: { width: this.canvas.width, height: this.canvas.height },
      format: this.format,
      sampleCount: 4,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.msaaView = this.msaaTexture.createView();
  }

  onResize(width: number, height: number): void {
    this.camera.onResize(width, height);
    this.createDepthTexture();
    this.createMSAATexture();
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

    this.device.queue.writeBuffer(this.globalsBuffer, 0, globalsData);

    this.ocean.update(time);
    this.flock.update(dt);

    const encoder = this.device.createCommandEncoder({ label: 'Frame Encoder' });

    const renderPass = encoder.beginRenderPass({
      label: 'Main Render Pass',
      colorAttachments: [{
        view: this.msaaView,
        resolveTarget: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.45, g: 0.65, b: 0.85, a: 1.0 },
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
    this.flock.encode(renderPass);

    renderPass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
