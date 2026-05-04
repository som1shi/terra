import birdsShaderSource from '../shaders/birds.wgsl?raw';

const NUM_BIRDS    = 40;
const SPEED        = 50.0;
const MAX_TURN     = 1.2;
const TARGET_ALT   = 320.0;
const AVOID_RADIUS = 45.0;
const AVOID_NBRS   = 3;

const K_AVOID = 6.0;
const K_ALIGN = 1.2;
const K_COHES = 0.10;
const K_BOUND = 0.02;
const BOUNDARY_B = 4;
const NEIGHBOR_K = 7;

const NUM_GROUPS      = 3;
const GROUP_RING_R    = 650;
const GROUP_SPREAD    = 180;

const TERRAIN_GRID         = 512;
const TERRAIN_WORLD_HALF   = 2048.0;
const TERRAIN_HEIGHT_SCALE = 600.0;
const TERRAIN_SAMPLE_STEP  = 80.0;
const TERRAIN_LOOKAHEAD    = 320.0;
const K_TERR               = 9.0;
const MOUNTAIN_LOW         = 180.0;
const MOUNTAIN_HIGH        = 260.0;
const ISLAND_RADIUS        = 900.0;
const K_ISLAND             = 2.5;
const K_WANDER             = 0.35;
const WANDER_TURN          = 0.35;

export class Flock {
  private pos: Float32Array;
  private vel: Float32Array;

  private instanceData: Float32Array;
  private instanceBuffer!: GPUBuffer;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;

  private pipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;
  private heightmapData: Float32Array | null = null;
  private wander: Float32Array;
  private wanderRate: Float32Array;

  constructor(
    private device: GPUDevice,
    private format: GPUTextureFormat,
    private globalsBuffer: GPUBuffer,
    private sampleCount: number = 4,
    heightmapData: Float32Array | null = null,
  ) {
    this.heightmapData = heightmapData;
    this.pos          = new Float32Array(NUM_BIRDS * 3);
    this.vel          = new Float32Array(NUM_BIRDS * 3);
    this.instanceData = new Float32Array(NUM_BIRDS * 8);
    this.wander       = new Float32Array(NUM_BIRDS);
    this.wanderRate   = new Float32Array(NUM_BIRDS);
    this.init();
  }

  private init(): void {
    const headingBase = Math.random() * Math.PI * 2;
    const centers: [number, number, number][] = [];
    for (let g = 0; g < NUM_GROUPS; g++) {
      const spawnAngle = (g / NUM_GROUPS) * Math.PI * 2 + (Math.random() - 0.5) * 1.0;
      const r          = GROUP_RING_R + (Math.random() - 0.5) * 250;
      const heading    = headingBase + (g / NUM_GROUPS) * Math.PI * 2;
      centers.push([Math.cos(spawnAngle) * r, Math.sin(spawnAngle) * r, heading]);
    }

    const weights = Array.from({ length: NUM_GROUPS }, () => 0.5 + Math.random());
    const wTotal  = weights.reduce((s, w) => s + w, 0);

    const loneCount = Math.round(NUM_BIRDS * 0.15);

    for (let i = 0; i < NUM_BIRDS; i++) {
      const b = i * 3;

      if (i < loneCount) {
        const a = Math.random() * Math.PI * 2;
        const r = 300 + Math.random() * 500;
        this.pos[b]   = Math.cos(a) * r;
        this.pos[b+1] = TARGET_ALT + (Math.random() - 0.5) * 100;
        this.pos[b+2] = Math.sin(a) * r;
      } else {
        let pick = Math.random() * wTotal, g = 0;
        for (let acc = 0; g < NUM_GROUPS - 1; g++) {
          acc += weights[g];
          if (pick <= acc) break;
        }
        const [gx, gz, gHeading] = centers[g];
        this.pos[b]   = gx + (Math.random() - 0.5) * GROUP_SPREAD;
        this.pos[b+1] = TARGET_ALT + (Math.random() - 0.5) * 60;
        this.pos[b+2] = gz + (Math.random() - 0.5) * GROUP_SPREAD;

        const dir     = gHeading + (Math.random() - 0.5) * 0.6;
        this.vel[b]   = Math.cos(dir) * SPEED;
        this.vel[b+1] = (Math.random() - 0.5) * SPEED * 0.05;
        this.vel[b+2] = Math.sin(dir) * SPEED;
        continue;
      }

      const dir     = Math.random() * Math.PI * 2;
      this.vel[b]   = Math.cos(dir) * SPEED;
      this.vel[b+1] = (Math.random() - 0.5) * SPEED * 0.05;
      this.vel[b+2] = Math.sin(dir) * SPEED;
    }

    for (let i = 0; i < NUM_BIRDS; i++) {
      this.wander[i]     = Math.random() * Math.PI * 2;
      this.wanderRate[i] = (Math.random() - 0.5) * 2 * WANDER_TURN;
    }
  }

  async initGPU(): Promise<void> {
    this.createBuffers();
    await this.createPipeline();
    this.createBindGroup();
    this.uploadInstances();
  }

  private createBuffers(): void {
    const verts = new Float32Array([
       0.00,   0.08,  0.62,   0.00,
       0.00,   0.18,  0.35,   0.00,
      -0.10,   0.10,  0.28,   0.00,
       0.10,   0.10,  0.28,   0.00,
       0.00,  -0.06,  0.10,   0.00,
      -0.22,   0.10,  0.08,   0.00,
       0.22,   0.10,  0.08,   0.00,
      -0.55,   0.18,  0.00,   0.40,
       0.55,   0.18,  0.00,   0.40,
      -1.00,   0.28, -0.12,   1.00,
       1.00,   0.28, -0.12,   1.00,
       0.00,   0.06, -0.40,   0.00,
    ]);
    this.vertexBuffer = this.device.createBuffer({
      label: 'Bird Vertex Buffer',
      size:  verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, verts);

    const indices = new Uint16Array([
       0,  2,  1,
       0,  1,  3,
       1,  2,  4,
       1,  4,  3,
       2,  5,  4,
       3,  4,  6,
       2,  7,  5,
       5,  7, 11,
       7,  9, 11,
       3,  6,  8,
       6, 11,  8,
       8, 11, 10,
       4, 11,  5,
       4,  6, 11,
    ]);
    this.indexBuffer = this.device.createBuffer({
      label: 'Bird Index Buffer',
      size:  indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indexBuffer, 0, indices);

    this.instanceBuffer = this.device.createBuffer({
      label: 'Bird Instance Buffer',
      size:  NUM_BIRDS * 8 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  private async createPipeline(): Promise<void> {
    const module = this.device.createShaderModule({
      label: 'Birds Shader',
      code:  birdsShaderSource,
    });

    const bgl = this.device.createBindGroupLayout({
      label: 'Birds BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform'           } },
        { binding: 1, visibility: GPUShaderStage.VERTEX,                            buffer: { type: 'read-only-storage' } },
      ],
    });

    this.pipeline = await this.device.createRenderPipelineAsync({
      label:  'Birds Render Pipeline',
      layout: this.device.createPipelineLayout({ label: 'Birds Layout', bindGroupLayouts: [bgl] }),
      vertex: {
        module,
        entryPoint: 'vs_main',
        buffers: [{
          arrayStride: 16,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' as GPUVertexFormat },
          ],
        }],
      },
      fragment: {
        module,
        entryPoint: 'fs_main',
        targets: [{ format: this.format }],
      },
      primitive:    { topology: 'triangle-list', cullMode: 'none' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
      multisample:  { count: this.sampleCount },
    });
  }

  private createBindGroup(): void {
    const bgl = this.pipeline.getBindGroupLayout(0);
    this.bindGroup = this.device.createBindGroup({
      label: 'Birds BG',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.globalsBuffer   } },
        { binding: 1, resource: { buffer: this.instanceBuffer  } },
      ],
    });
  }

  private uploadInstances(): void {
    for (let i = 0; i < NUM_BIRDS; i++) {
      const pb = i * 3;
      const gb = i * 8;

      const vx = this.vel[pb], vy = this.vel[pb+1], vz = this.vel[pb+2];
      const spd = Math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-6;

      this.instanceData[gb]   = this.pos[pb];
      this.instanceData[gb+1] = this.pos[pb+1];
      this.instanceData[gb+2] = this.pos[pb+2];
      this.instanceData[gb+3] = 0;
      this.instanceData[gb+4] = vx / spd;
      this.instanceData[gb+5] = vy / spd;
      this.instanceData[gb+6] = vz / spd;
      this.instanceData[gb+7] = 0;
    }
    this.device.queue.writeBuffer(
      this.instanceBuffer, 0,
      this.instanceData.buffer as ArrayBuffer,
      this.instanceData.byteOffset,
      this.instanceData.byteLength,
    );
  }

  private sampleTerrainHeight(x: number, z: number): number {
    if (!this.heightmapData) return 0;
    const uvX = (x + TERRAIN_WORLD_HALF) / (TERRAIN_WORLD_HALF * 2);
    const uvZ = (z + TERRAIN_WORLD_HALF) / (TERRAIN_WORLD_HALF * 2);
    const px  = Math.max(0, Math.min(TERRAIN_GRID - 1, Math.floor(uvX * TERRAIN_GRID)));
    const pz  = Math.max(0, Math.min(TERRAIN_GRID - 1, Math.floor(uvZ * TERRAIN_GRID)));
    return this.heightmapData[pz * TERRAIN_GRID + px] * TERRAIN_HEIGHT_SCALE;
  }

  update(dt: number): void {
    const step = Math.min(dt, 0.033);
    const N    = NUM_BIRDS;
    const pos  = this.pos;
    const vel  = this.vel;

    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < N; i++) {
      cx += pos[i*3]; cy += pos[i*3+1]; cz += pos[i*3+2];
    }
    cx /= N; cy /= N; cz /= N;

    const newVel = new Float32Array(N * 3);

    for (let i = 0; i < N; i++) {
      const b  = i * 3;
      const px = pos[b], py = pos[b+1], pz = pos[b+2];
      const vx = vel[b], vy = vel[b+1], vz = vel[b+2];
      const spd = Math.sqrt(vx*vx + vy*vy + vz*vz) + 1e-6;
      const fx = vx/spd, fy = vy/spd, fz = vz/spd;

      const dists: { d2: number; j: number }[] = [];
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const dx = pos[j*3]-px, dy = pos[j*3+1]-py, dz = pos[j*3+2]-pz;
        dists.push({ d2: dx*dx + dy*dy + dz*dz, j });
      }
      dists.sort((a, b) => a.d2 - b.d2);
      const nbrs = dists.slice(0, NEIGHBOR_K);
      const ni   = nbrs.length;

      let avX = 0, avY = 0, avZ = 0;
      const avoidCount = Math.min(AVOID_NBRS, ni);
      for (let k = 0; k < avoidCount; k++) {
        const { d2, j } = nbrs[k];
        const dist = Math.sqrt(d2) + 1e-6;
        const strength = Math.max(0, 1 - dist / AVOID_RADIUS);
        if (strength <= 0) break;
        avX += -(pos[j*3]   - px) / dist * strength;
        avY += -(pos[j*3+1] - py) / dist * strength;
        avZ += -(pos[j*3+2] - pz) / dist * strength;
      }

      let alX = 0, alY = 0, alZ = 0;
      for (const { j } of nbrs) {
        alX += vel[j*3]; alY += vel[j*3+1]; alZ += vel[j*3+2];
      }
      if (ni > 0) {
        const len = Math.sqrt(alX*alX + alY*alY + alZ*alZ) + 1e-6;
        alX /= len; alY /= len; alZ /= len;
      }

      let coX = 0, coY = 0, coZ = 0;
      for (const { j } of nbrs) {
        coX += pos[j*3]-px; coY += pos[j*3+1]-py; coZ += pos[j*3+2]-pz;
      }
      if (ni > 0) {
        const len = Math.sqrt(coX*coX + coY*coY + coZ*coZ) + 1e-6;
        coX /= len; coY /= len; coZ /= len;
      }

      const isolation = Math.max(0, 1 - ni / BOUNDARY_B);
      let bnX = cx-px, bnY = cy-py, bnZ = cz-pz;
      const bLen = Math.sqrt(bnX*bnX + bnY*bnY + bnZ*bnZ) + 1e-6;
      bnX /= bLen; bnY /= bLen; bnZ /= bLen;

      const terrainH = this.sampleTerrainHeight(px, pz);
      const hR  = this.sampleTerrainHeight(px + TERRAIN_SAMPLE_STEP, pz);
      const hL  = this.sampleTerrainHeight(px - TERRAIN_SAMPLE_STEP, pz);
      const hFw = this.sampleTerrainHeight(px, pz + TERRAIN_SAMPLE_STEP);
      const hBk = this.sampleTerrainHeight(px, pz - TERRAIN_SAMPLE_STEP);
      const LD = TERRAIN_LOOKAHEAD;
      const rightX = -fz, rightZ = fx;
      const hAheadC = this.sampleTerrainHeight(px + fx*LD,                       pz + fz*LD);
      const hAheadL = this.sampleTerrainHeight(px + fx*LD*0.7 - rightX*LD*0.7,   pz + fz*LD*0.7 - rightZ*LD*0.7);
      const hAheadR = this.sampleTerrainHeight(px + fx*LD*0.7 + rightX*LD*0.7,   pz + fz*LD*0.7 + rightZ*LD*0.7);
      const hAhead  = Math.max(hAheadC, hAheadL, hAheadR);
      const gradX = hR - hL;
      const gradZ = hFw - hBk;
      const gLen  = Math.sqrt(gradX*gradX + gradZ*gradZ) + 1e-6;
      const maxNearby = Math.max(terrainH, hAhead, hR, hL, hFw, hBk);

      const sideBias = Math.sign(hAheadL - hAheadR);
      const biasStr  = Math.min(1.0, Math.abs(hAheadL - hAheadR) / 80);

      const mBlend = Math.max(0, Math.min(1, (maxNearby - MOUNTAIN_LOW) / (MOUNTAIN_HIGH - MOUNTAIN_LOW)));
      const smoothT = mBlend * mBlend * (3 - 2 * mBlend);

      const overTarget = Math.max(TARGET_ALT, maxNearby + 60);
      const altErr     = (1 - smoothT) * (overTarget - py) * 0.012
                       +      smoothT  * (TARGET_ALT  - py) * 0.005;

      const horizThreat = smoothT * Math.max(0, maxNearby - py + 60);
      const hScale      = Math.min(2.5, horizThreat / 60);
      const trrX = (-(gradX / gLen) + rightX * sideBias * biasStr) * hScale;
      const trrZ = (-(gradZ / gLen) + rightZ * sideBias * biasStr) * hScale;

      const distFromIsland = Math.sqrt(px*px + pz*pz);
      const islandPull = Math.max(0, (distFromIsland - ISLAND_RADIUS) / ISLAND_RADIUS);
      const islandX = (-px / (distFromIsland + 1e-6)) * islandPull;
      const islandZ = (-pz / (distFromIsland + 1e-6)) * islandPull;

      const wa  = this.wander[i];
      const wdX = Math.cos(wa) * K_WANDER;
      const wdZ = Math.sin(wa) * K_WANDER;

      const yDamp = 0.15;
      let dx = K_AVOID*avX + K_ALIGN*alX + K_COHES*coX + K_BOUND*isolation*bnX + K_TERR*trrX + K_ISLAND*islandX + wdX;
      let dy = yDamp*(K_AVOID*avY + K_ALIGN*alY + K_COHES*coY + K_BOUND*isolation*bnY) + altErr;
      let dz = K_AVOID*avZ + K_ALIGN*alZ + K_COHES*coZ + K_BOUND*isolation*bnZ + K_TERR*trrZ + K_ISLAND*islandZ + wdZ;

      const dlen = Math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
      dx /= dlen; dy /= dlen; dz /= dlen;

      const dotFD = Math.max(-1, Math.min(1, fx*dx + fy*dy + fz*dz));
      const angle = Math.acos(dotFD);
      const t     = angle < 1e-4 ? 1 : Math.min(MAX_TURN * step, angle) / angle;

      let nfx = fx + t*(dx - fx);
      let nfy = fy + t*(dy - fy);
      let nfz = fz + t*(dz - fz);

      const emergencyUp = Math.max(0, terrainH + 100 - py) / 100;
      nfy += emergencyUp * 0.5;

      const nfl = Math.sqrt(nfx*nfx + nfy*nfy + nfz*nfz) + 1e-6;
      nfx /= nfl; nfy /= nfl; nfz /= nfl;

      newVel[b]   = nfx * SPEED;
      newVel[b+1] = nfy * SPEED;
      newVel[b+2] = nfz * SPEED;

      this.wander[i] += this.wanderRate[i] * step;
    }

    const WRAP = 1900;
    for (let i = 0; i < N; i++) {
      const b = i * 3;
      pos[b]   += newVel[b]   * step;
      pos[b+1] += newVel[b+1] * step;
      pos[b+2] += newVel[b+2] * step;

      if (pos[b]   >  WRAP) pos[b]   -= WRAP * 2;
      if (pos[b]   < -WRAP) pos[b]   += WRAP * 2;
      if (pos[b+2] >  WRAP) pos[b+2] -= WRAP * 2;
      if (pos[b+2] < -WRAP) pos[b+2] += WRAP * 2;
    }

    vel.set(newVel);
    this.uploadInstances();
  }

  encode(pass: GPURenderPassEncoder): void {
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.setVertexBuffer(0, this.vertexBuffer);
    pass.setIndexBuffer(this.indexBuffer, 'uint16');
    pass.drawIndexed(42, NUM_BIRDS);
  }
}
