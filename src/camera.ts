function mat4Multiply(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) {
        sum += a[row + k * 4] * b[k + col * 4];
      }
      out[row + col * 4] = sum;
    }
  }
  return out;
}

function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1.0 / Math.tan(fovY / 2);
  const nf = 1.0 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = far * nf;
  m[11] = -1;
  m[14] = far * near * nf;
  return m;
}

function mat4LookAt(eye: Float32Array, center: Float32Array, up: Float32Array): Float32Array {
  const fx = center[0] - eye[0];
  const fy = center[1] - eye[1];
  const fz = center[2] - eye[2];
  const fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
  const fnx = fx / fl, fny = fy / fl, fnz = fz / fl;

  const sx = fny * up[2] - fnz * up[1];
  const sy = fnz * up[0] - fnx * up[2];
  const sz = fnx * up[1] - fny * up[0];
  const sl = Math.sqrt(sx * sx + sy * sy + sz * sz);
  const snx = sx / sl, sny = sy / sl, snz = sz / sl;

  const ux = sny * fnz - snz * fny;
  const uy = snz * fnx - snx * fnz;
  const uz = snx * fny - sny * fnx;

  const m = new Float32Array(16);
  m[0] = snx; m[4] = sny; m[8] = snz;
  m[1] = ux; m[5] = uy; m[9] = uz;
  m[2] = -fnx; m[6] = -fny; m[10] = -fnz;
  m[3] = 0; m[7] = 0; m[11] = 0;
  m[12] = -(snx * eye[0] + sny * eye[1] + snz * eye[2]);
  m[13] = -(ux * eye[0] + uy * eye[1] + uz * eye[2]);
  m[14] = fnx * eye[0] + fny * eye[1] + fnz * eye[2];
  m[15] = 1;
  return m;
}

function mat4Invert(m: Float32Array): Float32Array {
  const out = new Float32Array(16);
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10, b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30, b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-16) { out.fill(0); out[0] = out[5] = out[10] = out[15] = 1; return out; }
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

export class Camera {
  private pos: Float32Array = new Float32Array([0, 500, 800]);
  private yaw = Math.PI;
  private pitch = -0.35;
  private keys: Set<string> = new Set();
  private baseSpeed = 150;
  private speedMultiplier = 1.0;
  private isPointerLocked = false;
  private mouseDX = 0;
  private mouseDY = 0;
  private _viewMatrix: Float32Array = new Float32Array(16);
  private _projMatrix: Float32Array = new Float32Array(16);
  private _viewProjMatrix: Float32Array = new Float32Array(16);
  private aspect: number;
  private dirty = true;

  constructor(private canvas: HTMLCanvasElement) {
    this.aspect = canvas.width / canvas.height;
    this.setupInputListeners();
    this.updateMatrices();
  }

  private setupInputListeners(): void {
    window.addEventListener('keydown', (e) => {
      this.keys.add(e.code);
    });

    window.addEventListener('keyup', (e) => {
      this.keys.delete(e.code);
    });

    this.canvas.addEventListener('click', () => {
      this.canvas.requestPointerLock();
    });

    document.addEventListener('pointerlockchange', () => {
      this.isPointerLocked = document.pointerLockElement === this.canvas;
    });

    document.addEventListener('mousemove', (e) => {
      if (this.isPointerLocked) {
        this.mouseDX += e.movementX;
        this.mouseDY += e.movementY;
      }
    });

    this.canvas.addEventListener('wheel', (e) => {
      this.speedMultiplier *= e.deltaY > 0 ? 0.9 : 1.1;
      this.speedMultiplier = Math.max(0.1, Math.min(20, this.speedMultiplier));
    });
  }

  update(dt: number): void {
    const sensitivity = 0.002;
    const speed = this.baseSpeed * this.speedMultiplier * dt;

    if (this.isPointerLocked) {
      this.yaw -= this.mouseDX * sensitivity;
      this.pitch -= this.mouseDY * sensitivity;
      this.pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, this.pitch));
    }
    this.mouseDX = 0;
    this.mouseDY = 0;

    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);

    const forward = new Float32Array([cp * sy, sp, cp * cy]);
    const right = new Float32Array([cy, 0, -sy]);
    const rl = Math.sqrt(right[0] * right[0] + right[2] * right[2]);
    right[0] /= rl; right[2] /= rl;

    const sprintFactor = this.keys.has('ShiftLeft') || this.keys.has('ShiftRight') ? 4 : 1;
    const actualSpeed = speed * sprintFactor;

    let moved = false;

    if (this.keys.has('KeyW') || this.keys.has('ArrowUp')) {
      this.pos[0] += forward[0] * actualSpeed;
      this.pos[1] += forward[1] * actualSpeed;
      this.pos[2] += forward[2] * actualSpeed;
      moved = true;
    }
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown')) {
      this.pos[0] -= forward[0] * actualSpeed;
      this.pos[1] -= forward[1] * actualSpeed;
      this.pos[2] -= forward[2] * actualSpeed;
      moved = true;
    }
    if (this.keys.has('KeyA') || this.keys.has('ArrowRight')) {
      this.pos[0] += right[0] * actualSpeed;
      this.pos[2] += right[2] * actualSpeed;
      moved = true;
    }
    if (this.keys.has('KeyD') || this.keys.has('ArrowLeft')) {
      this.pos[0] -= right[0] * actualSpeed;
      this.pos[2] -= right[2] * actualSpeed;
      moved = true;
    }
    if (this.keys.has('KeyE')) {
      this.pos[1] += actualSpeed;
      moved = true;
    }
    if (this.keys.has('KeyQ')) {
      this.pos[1] -= actualSpeed;
      moved = true;
    }

    if (moved || this.isPointerLocked) {
      this.dirty = true;
    }

    if (this.dirty) {
      this.updateMatrices();
      this.dirty = false;
    }
  }

  private updateMatrices(): void {
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);

    const lookTarget = new Float32Array([
      this.pos[0] + cp * sy,
      this.pos[1] + sp,
      this.pos[2] + cp * cy,
    ]);

    const up = new Float32Array([0, 1, 0]);
    this._viewMatrix = mat4LookAt(this.pos, lookTarget, up);
    this._projMatrix = mat4Perspective(
      (60 * Math.PI) / 180,
      this.aspect,
      0.1,
      10000
    );
    this._viewProjMatrix = mat4Multiply(this._projMatrix, this._viewMatrix);
  }

  onResize(width: number, height: number): void {
    this.aspect = width / height;
    this.dirty = true;
    this.updateMatrices();
  }

  get viewProjMatrix(): Float32Array { return this._viewProjMatrix; }
  get viewMatrix(): Float32Array { return this._viewMatrix; }
  get projMatrix(): Float32Array { return this._projMatrix; }
  get position(): [number, number, number] {
    return [this.pos[0], this.pos[1], this.pos[2]];
  }

  getInverseViewProj(): Float32Array {
    return mat4Invert(this._viewProjMatrix);
  }
}
