function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(a: number, b: number, t: number): number {
  return a + t * (b - a);
}

function grad(hash: number, x: number, y: number): number {
  const h = hash & 7;
  const u = h < 4 ? x : y;
  const v = h < 4 ? y : x;
  return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

class PermutationTable {
  private p: Uint8Array;

  constructor(seed: number = 42) {
    this.p = new Uint8Array(512);
    const perm = new Uint8Array(256);
    for (let i = 0; i < 256; i++) perm[i] = i;

    let s = seed | 0;
    for (let i = 255; i > 0; i--) {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      const j = ((s >>> 0) % (i + 1));
      const tmp = perm[i];
      perm[i] = perm[j];
      perm[j] = tmp;
    }

    for (let i = 0; i < 512; i++) {
      this.p[i] = perm[i & 255];
    }
  }

  noise2D(x: number, y: number): number {
    const xi = Math.floor(x) & 255;
    const yi = Math.floor(y) & 255;
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);

    const u = fade(xf);
    const v = fade(yf);

    const aa = this.p[this.p[xi] + yi];
    const ab = this.p[this.p[xi] + yi + 1];
    const ba = this.p[this.p[xi + 1] + yi];
    const bb = this.p[this.p[xi + 1] + yi + 1];

    return lerp(
      lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u),
      lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u),
      v
    );
  }
}

export class Heightmap {
  private data: Float32Array<ArrayBuffer>;

  constructor(private width: number, private height: number, private seed: number = 137) {
    this.data = new Float32Array(width * height);
  }

  generate(): Float32Array<ArrayBuffer> {
    const perm = new PermutationTable(this.seed);

    const octaves = 7;
    const lacunarity = 2.0;
    const persistence = 0.40;
    const baseFrequency = 1.2;
    const sharpness = 1.5;

    const cx = this.width / 2;
    const cy = this.height / 2;
    const maxDist = Math.sqrt(cx * cx + cy * cy);

    let globalMin = Infinity;
    let globalMax = -Infinity;

    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        const nx = (x / this.width) * 4.0;
        const ny = (y / this.height) * 4.0;

        let val = 0;
        let amp = 1.0;
        let freq = baseFrequency;
        let weight = 1.0;

        for (let o = 0; o < octaves; o++) {
          const n = perm.noise2D(nx * freq, ny * freq);
          let signal = 1.0 - Math.abs(n);
          signal = Math.pow(signal, sharpness);
          val += amp * signal * weight;
          weight = Math.min(1.0, signal * 2.0);
          amp *= persistence;
          freq *= lacunarity;
        }

        const warpFreq = 0.55;
        const warpAmt = 0.38;
        const wx = perm.noise2D(nx * warpFreq + 3.7, ny * warpFreq + 1.4) * warpAmt * maxDist;
        const wy = perm.noise2D(nx * warpFreq + 8.1, ny * warpFreq + 6.2) * warpAmt * maxDist;
        const dx = (x - cx) + wx;
        const dy = (y - cy) + wy;
        const dist = Math.sqrt(dx * dx + dy * dy) / maxDist;
        const falloff = 1.0 - Math.pow(Math.max(0, dist * 1.2 - 0.1), 1.3);
        const smoothFalloff = Math.max(0, falloff);

        val *= smoothFalloff;
        val = Math.pow(val, 2.2);

        this.data[y * this.width + x] = val;

        if (val < globalMin) globalMin = val;
        if (val > globalMax) globalMax = val;
      }
    }

    const range = globalMax - globalMin;
    if (range > 0) {
      for (let i = 0; i < this.data.length; i++) {
        this.data[i] = ((this.data[i] - globalMin) / range) * 0.72;
      }
    }

    const w = this.width;
    const h = this.height;
    const tmp = new Float32Array(w * h);
    const K = [1, 4, 7, 4, 1, 4, 16, 26, 16, 4, 7, 26, 41, 26, 7, 4, 16, 26, 16, 4, 1, 4, 7, 4, 1];
    const Ksum = K.reduce((a, b) => a + b, 0);

    for (let pass = 0; pass < 1; pass++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          let sum = 0;
          for (let ky = 0; ky < 5; ky++) {
            for (let kx = 0; kx < 5; kx++) {
              const sx = Math.min(Math.max(x + kx - 2, 0), w - 1);
              const sy = Math.min(Math.max(y + ky - 2, 0), h - 1);
              sum += this.data[sy * w + sx] * K[ky * 5 + kx];
            }
          }
          tmp[y * w + x] = sum / Ksum;
        }
      }
      this.data.set(tmp);
    }

    return this.data;
  }

  getWidth(): number { return this.width; }
  getHeight(): number { return this.height; }
  getData(): Float32Array<ArrayBuffer> { return this.data; }
}
