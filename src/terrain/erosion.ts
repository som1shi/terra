import erosionShaderSource from '../shaders/erosion.wgsl?raw';
import flowSmoothSource from '../shaders/flow_smooth.wgsl?raw';

const SIZE = 512;
const OUTER_ITER = 300;
const REACCUM_EVERY = 25;
const ACCUM_FULL = 250;
const THERMAL1_ITER = 40;
const THERMAL2_ITER = 6;
const THERMAL3_ITER = 12;
const FLATTEN_ITER = 8;
const WG = 8;
const DISPATCH = SIZE / WG;

export class ErosionSystem {
  private sedTex!: GPUTexture;
  private flowBuf!: GPUBuffer;
  private accumTex!: GPUTexture;
  private smoothedAccumTex!: GPUTexture;

  private bgl!: GPUBindGroupLayout;
  private bglSmooth!: GPUBindGroupLayout;
  private pipelineMFD!: GPUComputePipeline;
  private pipelineAccumInit!: GPUComputePipeline;
  private pipelineAccumulate!: GPUComputePipeline;
  private pipelineSPE!: GPUComputePipeline;
  private pipelineTransport!: GPUComputePipeline;
  private pipelineDeposit!: GPUComputePipeline;
  private pipelineUplift!: GPUComputePipeline;
  private pipelineThermal1!: GPUComputePipeline;
  private pipelineThermal2!: GPUComputePipeline;
  private pipelineThermal3!: GPUComputePipeline;
  private pipelineFlatten!: GPUComputePipeline;
  private pipelineBake!: GPUComputePipeline;
  private pipelineSmooth!: GPUComputePipeline;

  constructor(private device: GPUDevice) {
    this.createTextures();
    this.createPipelines();
  }

  private createTextures(): void {
    const baseR = {
      size: { width: SIZE, height: SIZE },
      usage: GPUTextureUsage.STORAGE_BINDING,
      format: 'r32float' as GPUTextureFormat,
    };
    this.sedTex = this.device.createTexture({ label: 'Sediment', ...baseR });
    this.accumTex = this.device.createTexture({
      label: 'Drainage Area',
      size: { width: SIZE, height: SIZE },
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      format: 'r32float',
    });
    this.smoothedAccumTex = this.device.createTexture({
      label: 'Smoothed Drainage Area',
      size: { width: SIZE, height: SIZE },
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      format: 'r32float',
    });
    this.flowBuf = this.device.createBuffer({
      label: 'MFD Flow Buffer',
      size: SIZE * SIZE * 8 * 4,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  private createPipelines(): void {
    this.bgl = this.device.createBindGroupLayout({
      label: 'Erosion BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-write', format: 'r32float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-write', format: 'r32float' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-write', format: 'r32float' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    const layout = this.device.createPipelineLayout({
      label: 'Erosion Pipeline Layout',
      bindGroupLayouts: [this.bgl],
    });

    const module = this.device.createShaderModule({
      label: 'Erosion Shader',
      code: erosionShaderSource,
    });

    const make = (ep: string, label: string): GPUComputePipeline =>
      this.device.createComputePipeline({ label, layout, compute: { module, entryPoint: ep } });

    this.pipelineMFD = make('mfd_direction', 'MFD Direction Pipeline');
    this.pipelineAccumInit = make('accum_init', 'Accum Init Pipeline');
    this.pipelineAccumulate = make('flow_accumulate', 'Flow Accumulate Pipeline');
    this.pipelineSPE = make('spe_erode', 'SPE Erode Pipeline');
    this.pipelineTransport = make('sed_transport', 'Sediment Transport Pipeline');
    this.pipelineDeposit = make('sed_deposit', 'Sediment Deposit Pipeline');
    this.pipelineUplift = make('uplift_bedrock', 'Uplift Pipeline');
    this.pipelineThermal1 = make('thermal_zone1', 'Thermal Zone1 Pipeline');
    this.pipelineThermal2 = make('thermal_zone2', 'Thermal Zone2 Pipeline');
    this.pipelineThermal3 = make('thermal_zone3', 'Thermal Zone3 Pipeline');
    this.pipelineFlatten = make('flatten_lowlands', 'Flatten Lowlands Pipeline');
    this.pipelineBake = make('bake_sediment', 'Bake Sediment Pipeline');

    this.bglSmooth = this.device.createBindGroupLayout({
      label: 'Flow Smooth BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'r32float' } },
      ],
    });
    this.pipelineSmooth = this.device.createComputePipeline({
      label: 'Flow Smooth Pipeline',
      layout: this.device.createPipelineLayout({
        label: 'Flow Smooth Layout',
        bindGroupLayouts: [this.bglSmooth],
      }),
      compute: {
        module: this.device.createShaderModule({ label: 'Flow Smooth Shader', code: flowSmoothSource }),
        entryPoint: 'smooth_flow',
      },
    });
  }

  getAccumTex(): GPUTexture { return this.accumTex; }
  getSmoothedAccumTex(): GPUTexture { return this.smoothedAccumTex; }

  run(encoder: GPUCommandEncoder, heightTex: GPUTexture): void {
    const bg = this.device.createBindGroup({
      label: 'Erosion Bind Group',
      layout: this.bgl,
      entries: [
        { binding: 0, resource: heightTex.createView() },
        { binding: 1, resource: this.sedTex.createView() },
        { binding: 2, resource: this.accumTex.createView() },
        { binding: 3, resource: { buffer: this.flowBuf } },
      ],
    });

    for (let outer = 0; outer < OUTER_ITER; outer++) {
      if (outer % REACCUM_EVERY === 0) {
        this.dispatch(encoder, this.pipelineMFD, bg, 'MFD Direction');
        this.dispatch(encoder, this.pipelineAccumInit, bg, 'Accum Init');
        for (let k = 0; k < ACCUM_FULL; k++) {
          this.dispatch(encoder, this.pipelineAccumulate, bg, 'Accum');
        }
      }
      this.dispatch(encoder, this.pipelineSPE, bg, 'SPE Erode');
      this.dispatch(encoder, this.pipelineTransport, bg, 'Sed Transport');
      this.dispatch(encoder, this.pipelineDeposit, bg, 'Sed Deposit');
      this.dispatch(encoder, this.pipelineUplift, bg, 'Uplift');
    }

    for (let i = 0; i < THERMAL1_ITER; i++) {
      this.dispatch(encoder, this.pipelineThermal1, bg, 'Thermal1');
    }
    for (let i = 0; i < THERMAL2_ITER; i++) {
      this.dispatch(encoder, this.pipelineThermal2, bg, 'Thermal2');
    }
    for (let i = 0; i < THERMAL3_ITER; i++) {
      this.dispatch(encoder, this.pipelineThermal3, bg, 'Thermal3');
    }

    for (let i = 0; i < FLATTEN_ITER; i++) {
      this.dispatch(encoder, this.pipelineFlatten, bg, 'Flatten');
    }

    this.dispatch(encoder, this.pipelineBake, bg, 'Bake Sediment');

    const bgSmooth = this.device.createBindGroup({
      label: 'Flow Smooth BG',
      layout: this.bglSmooth,
      entries: [
        { binding: 0, resource: this.accumTex.createView() },
        { binding: 1, resource: this.smoothedAccumTex.createView() },
      ],
    });
    const smoothPass = encoder.beginComputePass({ label: 'Flow Smooth' });
    smoothPass.setPipeline(this.pipelineSmooth);
    smoothPass.setBindGroup(0, bgSmooth);
    smoothPass.dispatchWorkgroups(DISPATCH, DISPATCH);
    smoothPass.end();
  }

  private dispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bg: GPUBindGroup,
    label: string,
  ): void {
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(DISPATCH, DISPATCH);
    pass.end();
  }
}
