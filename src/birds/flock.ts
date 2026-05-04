import birdsShaderSource from '../shaders/birds.wgsl?raw';

const NUM_BIRDS    = 40;
const SPEED        = 50.0;  // world units / second
const MAX_TURN     = 1.2;   // radians / second — slower turns, more stable flight
const TARGET_ALT   = 320.0; // desired flight altitude
const AVOID_RADIUS = 45.0;  // personal-space bubble radius (world units)
const AVOID_NBRS   = 3;     // how many nearest birds to avoid (not just 1)

// Social weights
const K_AVOID = 6.0;   // strong short-range repulsion
const K_ALIGN = 1.2;   // match neighbor directions
const K_COHES = 0.10;  // gentle cohesion — loose natural groups
const K_BOUND = 0.02;  // nearly off — global centroid pull was merging groups
const BOUNDARY_B = 4;  // fades out quickly so most birds are unaffected
const NEIGHBOR_K = 7;  // topological neighbors (from Flock2 paper)

// Group spawn config
const NUM_GROUPS      = 3;
const GROUP_RING_R    = 650;  // approx radius around island center
const GROUP_SPREAD    = 180;  // wide spread so groups feel organic, not rigid

// Terrain avoidance
const TERRAIN_GRID         = 512;
const TERRAIN_WORLD_HALF   = 2048.0;
const TERRAIN_HEIGHT_SCALE = 600.0;
const TERRAIN_SAMPLE_STEP  = 80.0;   // gradient sample offset (world units)
const TERRAIN_LOOKAHEAD    = 320.0;  // lookahead distance — react to terrain earlier
const K_TERR               = 9.0;    // horizontal terrain steering weight
const MOUNTAIN_LOW         = 180.0;  // below this: fly over
const MOUNTAIN_HIGH        = 260.0;  // above this: go around; blend in between
const ISLAND_RADIUS        = 900.0;  // birds beyond this distance get pulled back
const K_ISLAND             = 2.5;    // island return force strength
const K_WANDER             = 0.35;   // per-bird wander nudge — causes merged flocks to split
const WANDER_TURN          = 0.35;   // rad/s max drift of each bird's wander angle

export class Flock {
  private pos: Float32Array;       // [x,y,z] * NUM_BIRDS
  private vel: Float32Array;       // [vx,vy,vz] * NUM_BIRDS

  private instanceData: Float32Array; // 2 vec4s per bird = 8 floats
  private instanceBuffer!: GPUBuffer;
  private vertexBuffer!: GPUBuffer;
  private indexBuffer!: GPUBuffer;

  private pipeline!: GPURenderPipeline;
  private bindGroup!: GPUBindGroup;
  private heightmapData: Float32Array | null = null;
  private wander: Float32Array;     // per-bird slowly-drifting preferred heading (XZ angle)
  private wanderRate: Float32Array; // per-bird wander rotation rate

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
    // Groups get absolute headings 120° apart from a random base —
    // guarantees no two groups fly in the same direction regardless of spawn position
    const headingBase = Math.random() * Math.PI * 2;
    const centers: [number, number, number][] = []; // [x, z, heading]
    for (let g = 0; g < NUM_GROUPS; g++) {
      const spawnAngle = (g / NUM_GROUPS) * Math.PI * 2 + (Math.random() - 0.5) * 1.0;
      const r          = GROUP_RING_R + (Math.random() - 0.5) * 250;
      const heading    = headingBase + (g / NUM_GROUPS) * Math.PI * 2; // 120° apart, absolute
      centers.push([Math.cos(spawnAngle) * r, Math.sin(spawnAngle) * r, heading]);
    }

    // Random unequal group weights so sizes vary naturally
    const weights = Array.from({ length: NUM_GROUPS }, () => 0.5 + Math.random());
    const wTotal  = weights.reduce((s, w) => s + w, 0);

    // ~15% of birds spawn as loners at arbitrary positions
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
        // Pick group by weighted random
        let pick = Math.random() * wTotal, g = 0;
        for (let acc = 0; g < NUM_GROUPS - 1; g++) {
          acc += weights[g];
          if (pick <= acc) break;
        }
        const [gx, gz, gHeading] = centers[g];
        this.pos[b]   = gx + (Math.random() - 0.5) * GROUP_SPREAD;
        this.pos[b+1] = TARGET_ALT + (Math.random() - 0.5) * 60;
        this.pos[b+2] = gz + (Math.random() - 0.5) * GROUP_SPREAD;

        // Each group has its own heading with small per-bird variation
        const dir     = gHeading + (Math.random() - 0.5) * 0.6;
        this.vel[b]   = Math.cos(dir) * SPEED;
        this.vel[b+1] = (Math.random() - 0.5) * SPEED * 0.05;
        this.vel[b+2] = Math.sin(dir) * SPEED;
        continue;
      }

      // Loner: random heading
      const dir     = Math.random() * Math.PI * 2;
      this.vel[b]   = Math.cos(dir) * SPEED;
      this.vel[b+1] = (Math.random() - 0.5) * SPEED * 0.05;
      this.vel[b+2] = Math.sin(dir) * SPEED;
    }

    // Each bird gets a unique wander angle and a slow random drift rate
    for (let i = 0; i < NUM_BIRDS; i++) {
      this.wander[i]     = Math.random() * Math.PI * 2;
      this.wanderRate[i] = (Math.random() - 0.5) * 2 * WANDER_TURN;
    }
  }

  async initGPU(): Promise<void> {
    this.createBuffers();
    await this.createPipeline();
    this.createBindGroup();
    this.uploadInstances(); // initial upload
  }

  private createBuffers(): void {
    // 12-vertex bird, local space: +Z forward, +X right, +Y up.
    // w = flap weight (0 = rigid body, 0.4 = elbow, 1.0 = wing tip).
    // Wings have a ~16° dihedral (tips elevated) so the upper wing surface
    // faces a horizontal viewer instead of being edge-on.
    const verts = new Float32Array([
    //    x       y       z     flapW
       0.00,   0.08,  0.62,   0.00,  //  v0  nose tip
       0.00,   0.18,  0.35,   0.00,  //  v1  head top        ← body height visible from front
      -0.10,   0.10,  0.28,   0.00,  //  v2  head left
       0.10,   0.10,  0.28,   0.00,  //  v3  head right
       0.00,  -0.06,  0.10,   0.00,  //  v4  belly           ← visible from below/side
      -0.22,   0.10,  0.08,   0.00,  //  v5  left wing root
       0.22,   0.10,  0.08,   0.00,  //  v6  right wing root
      -0.55,   0.18,  0.00,   0.40,  //  v7  left elbow      ← dihedral begins
       0.55,   0.18,  0.00,   0.40,  //  v8  right elbow
      -1.00,   0.28, -0.12,   1.00,  //  v9  left tip        ← max dihedral
       1.00,   0.28, -0.12,   1.00,  //  v10 right tip
       0.00,   0.06, -0.40,   0.00,  //  v11 tail
    ]);
    this.vertexBuffer = this.device.createBuffer({
      label: 'Bird Vertex Buffer',
      size:  verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.vertexBuffer, 0, verts);

    // 14 triangles = 42 indices
    const indices = new Uint16Array([
      // Head / body (visible from front and sides)
       0,  2,  1,   //  nose → head, left face
       0,  1,  3,   //  nose → head, right face
       1,  2,  4,   //  head → belly, left
       1,  4,  3,   //  head → belly, right
      // Wing-root connections
       2,  5,  4,   //  left  body → wing root
       3,  4,  6,   //  right body → wing root
      // Left wing (upper surface faces upward/outward due to dihedral)
       2,  7,  5,   //  left  inner front
       5,  7, 11,   //  left  inner rear
       7,  9, 11,   //  left  outer
      // Right wing
       3,  6,  8,   //  right inner front
       6, 11,  8,   //  right inner rear
       8, 11, 10,   //  right outer
      // Lower body / tail
       4, 11,  5,   //  belly → tail, left
       4,  6, 11,   //  belly → tail, right
    ]);
    this.indexBuffer = this.device.createBuffer({
      label: 'Bird Index Buffer',
      size:  indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.indexBuffer, 0, indices);

    // Instance buffer: 2 vec4f per bird (pos + fwd)
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

      // pos vec4
      this.instanceData[gb]   = this.pos[pb];
      this.instanceData[gb+1] = this.pos[pb+1];
      this.instanceData[gb+2] = this.pos[pb+2];
      this.instanceData[gb+3] = 0;
      // fwd vec4 (normalized velocity)
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

    // Flock centroid for boundary term
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

      // Find NEIGHBOR_K nearest neighbors (simple O(N) scan)
      const dists: { d2: number; j: number }[] = [];
      for (let j = 0; j < N; j++) {
        if (j === i) continue;
        const dx = pos[j*3]-px, dy = pos[j*3+1]-py, dz = pos[j*3+2]-pz;
        dists.push({ d2: dx*dx + dy*dy + dz*dz, j });
      }
      dists.sort((a, b) => a.d2 - b.d2);
      const nbrs = dists.slice(0, NEIGHBOR_K);
      const ni   = nbrs.length;

      // Avoidance: push away from nearest AVOID_NBRS birds within AVOID_RADIUS.
      // Linear falloff so strength is 1 at contact, 0 at AVOID_RADIUS.
      let avX = 0, avY = 0, avZ = 0;
      const avoidCount = Math.min(AVOID_NBRS, ni);
      for (let k = 0; k < avoidCount; k++) {
        const { d2, j } = nbrs[k];
        const dist = Math.sqrt(d2) + 1e-6;
        const strength = Math.max(0, 1 - dist / AVOID_RADIUS);
        if (strength <= 0) break; // sorted by dist, so no closer birds after this
        avX += -(pos[j*3]   - px) / dist * strength;
        avY += -(pos[j*3+1] - py) / dist * strength;
        avZ += -(pos[j*3+2] - pz) / dist * strength;
      }

      // Alignment: match average neighbor velocity direction
      let alX = 0, alY = 0, alZ = 0;
      for (const { j } of nbrs) {
        alX += vel[j*3]; alY += vel[j*3+1]; alZ += vel[j*3+2];
      }
      if (ni > 0) {
        const len = Math.sqrt(alX*alX + alY*alY + alZ*alZ) + 1e-6;
        alX /= len; alY /= len; alZ /= len;
      }

      // Cohesion: toward neighborhood centroid
      let coX = 0, coY = 0, coZ = 0;
      for (const { j } of nbrs) {
        coX += pos[j*3]-px; coY += pos[j*3+1]-py; coZ += pos[j*3+2]-pz;
      }
      if (ni > 0) {
        const len = Math.sqrt(coX*coX + coY*coY + coZ*coZ) + 1e-6;
        coX /= len; coY /= len; coZ /= len;
      }

      // Boundary: peripheral birds turn toward flock center
      const isolation = Math.max(0, 1 - ni / BOUNDARY_B);
      let bnX = cx-px, bnY = cy-py, bnZ = cz-pz;
      const bLen = Math.sqrt(bnX*bnX + bnY*bnY + bnZ*bnZ) + 1e-6;
      bnX /= bLen; bnY /= bLen; bnZ /= bLen;

      // Terrain avoidance: sample gradient + 3-way lookahead (ahead, left, right)
      const terrainH = this.sampleTerrainHeight(px, pz);
      const hR  = this.sampleTerrainHeight(px + TERRAIN_SAMPLE_STEP, pz);
      const hL  = this.sampleTerrainHeight(px - TERRAIN_SAMPLE_STEP, pz);
      const hFw = this.sampleTerrainHeight(px, pz + TERRAIN_SAMPLE_STEP);
      const hBk = this.sampleTerrainHeight(px, pz - TERRAIN_SAMPLE_STEP);
      const LD = TERRAIN_LOOKAHEAD;
      const rightX = -fz, rightZ = fx; // XZ perpendicular to forward
      const hAheadC = this.sampleTerrainHeight(px + fx*LD,                       pz + fz*LD);
      const hAheadL = this.sampleTerrainHeight(px + fx*LD*0.7 - rightX*LD*0.7,   pz + fz*LD*0.7 - rightZ*LD*0.7);
      const hAheadR = this.sampleTerrainHeight(px + fx*LD*0.7 + rightX*LD*0.7,   pz + fz*LD*0.7 + rightZ*LD*0.7);
      const hAhead  = Math.max(hAheadC, hAheadL, hAheadR);
      const gradX = hR - hL;
      const gradZ = hFw - hBk;
      const gLen  = Math.sqrt(gradX*gradX + gradZ*gradZ) + 1e-6;
      const maxNearby = Math.max(terrainH, hAhead, hR, hL, hFw, hBk);

      // Directional bias: lean toward whichever side has lower terrain ahead
      const sideBias = Math.sign(hAheadL - hAheadR); // +1 = steer right, -1 = steer left
      const biasStr  = Math.min(1.0, Math.abs(hAheadL - hAheadR) / 80);

      // Blend: 0 = fly over (low), 1 = go around (high)
      const mBlend = Math.max(0, Math.min(1, (maxNearby - MOUNTAIN_LOW) / (MOUNTAIN_HIGH - MOUNTAIN_LOW)));
      const smoothT = mBlend * mBlend * (3 - 2 * mBlend);

      // Fly-over component
      const overTarget = Math.max(TARGET_ALT, maxNearby + 60);
      const altErr     = (1 - smoothT) * (overTarget - py) * 0.012
                       +      smoothT  * (TARGET_ALT  - py) * 0.005;

      // Go-around component: gradient repulsion + directional bias toward clearer path
      const horizThreat = smoothT * Math.max(0, maxNearby - py + 60);
      const hScale      = Math.min(2.5, horizThreat / 60);
      const trrX = (-(gradX / gLen) + rightX * sideBias * biasStr) * hScale;
      const trrZ = (-(gradZ / gLen) + rightZ * sideBias * biasStr) * hScale;

      // Island boundary: pull birds back toward (0,0) when they stray too far out to sea
      const distFromIsland = Math.sqrt(px*px + pz*pz);
      const islandPull = Math.max(0, (distFromIsland - ISLAND_RADIUS) / ISLAND_RADIUS);
      const islandX = (-px / (distFromIsland + 1e-6)) * islandPull;
      const islandZ = (-pz / (distFromIsland + 1e-6)) * islandPull;

      // Per-bird wander: slow unique drift that causes merged flocks to re-split over time
      const wa  = this.wander[i];
      const wdX = Math.cos(wa) * K_WANDER;
      const wdZ = Math.sin(wa) * K_WANDER;

      // Sum into desired direction — vertical boid forces scaled down to keep flight level
      const yDamp = 0.15;
      let dx = K_AVOID*avX + K_ALIGN*alX + K_COHES*coX + K_BOUND*isolation*bnX + K_TERR*trrX + K_ISLAND*islandX + wdX;
      let dy = yDamp*(K_AVOID*avY + K_ALIGN*alY + K_COHES*coY + K_BOUND*isolation*bnY) + altErr;
      let dz = K_AVOID*avZ + K_ALIGN*alZ + K_COHES*coZ + K_BOUND*isolation*bnZ + K_TERR*trrZ + K_ISLAND*islandZ + wdZ;

      const dlen = Math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-6;
      dx /= dlen; dy /= dlen; dz /= dlen;

      // Turn current forward toward desired direction, clamped to MAX_TURN*dt
      const dotFD = Math.max(-1, Math.min(1, fx*dx + fy*dy + fz*dz));
      const angle = Math.acos(dotFD);
      const t     = angle < 1e-4 ? 1 : Math.min(MAX_TURN * step, angle) / angle;

      let nfx = fx + t*(dx - fx);
      let nfy = fy + t*(dy - fy);
      let nfz = fz + t*(dz - fz);

      // Emergency lift applied after turn — can't be diluted by horizontal avoidance forces
      const emergencyUp = Math.max(0, terrainH + 100 - py) / 100;
      nfy += emergencyUp * 0.5;

      const nfl = Math.sqrt(nfx*nfx + nfy*nfy + nfz*nfz) + 1e-6;
      nfx /= nfl; nfy /= nfl; nfz /= nfl;

      newVel[b]   = nfx * SPEED;
      newVel[b+1] = nfy * SPEED;
      newVel[b+2] = nfz * SPEED;

      // Slowly rotate this bird's wander direction
      this.wander[i] += this.wanderRate[i] * step;
    }

    // Integrate positions, wrap horizontally
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
