struct Globals {
    viewProj      : mat4x4f,
    invViewProj   : mat4x4f,
    sunDir        : vec3f,
    _pad0         : f32,
    cameraPos     : vec3f,
    _pad1         : f32,
    time          : f32,
    timeOfDay     : f32,
    seaLevel      : f32,
    seed          : f32,
    resolution    : vec2f,
    moonIntensity : f32,
    _pad2         : f32,
    moonDir       : vec3f,
    _pad3         : f32,
};

const N  : u32 = 128u;
const PI : f32 = 3.14159265358979323846f;

@group(0) @binding(0) var<uniform>             globals : Globals;
@group(0) @binding(1) var<storage, read>       h0Buf   : array<vec4f>;
@group(0) @binding(2) var<storage, read>       freqBuf : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> specBuf : array<vec4f>;
@group(0) @binding(4) var                      oceanOut: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn spectrum_update(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= N || gid.y >= N) { return; }
    let idx  = gid.y * N + gid.x;
    let h0   = h0Buf[idx];
    let freq = freqBuf[idx];

    let wt = freq.x * globals.time;
    let cw = cos(wt);
    let sw = sin(wt);

    let e1r = h0.x * cw - h0.y * sw;
    let e1i = h0.x * sw + h0.y * cw;
    let e2r = h0.z * cw + h0.w * sw;
    let e2i = h0.w * cw - h0.z * sw;
    let htRe = e1r + e2r;
    let htIm = e1i + e2i;

    let kx = freq.y; let kz = freq.z;
    specBuf[idx * 2u]      = vec4f(htRe, htIm, -kx * htIm, kx * htRe);
    specBuf[idx * 2u + 1u] = vec4f(-kz * htIm, kz * htRe, 0.0, 0.0);
}

var<workgroup> smA: array<vec4f, 128>;
var<workgroup> smB: array<vec4f, 128>;

fn bitrev7(v: u32) -> u32 {
    var x = v & 0x7fu;
    x = ((x >> 1u) & 0x55u) | ((x & 0x55u) << 1u);
    x = ((x >> 2u) & 0x33u) | ((x & 0x33u) << 2u);
    x = ((x >> 4u) & 0x0fu) | ((x & 0x0fu) << 4u);
    return x >> 1u;
}

fn butterfly_stage(tid: u32, s: u32) {
    let half_m = 1u << (s - 1u);
    let j      = tid % half_m;
    let top    = (tid / half_m) * (half_m << 1u) + j;
    let bot    = top + half_m;

    let angle = 2.0 * PI * f32(j) / f32(half_m << 1u);
    let wr = cos(angle);
    let wi = sin(angle);

    let ta = smA[top]; let ba = smA[bot];
    let tb = smB[top]; let bb = smB[bot];

    let wRe0 = ba.x * wr - ba.y * wi;  let wIm0 = ba.x * wi + ba.y * wr;
    let wRe1 = ba.z * wr - ba.w * wi;  let wIm1 = ba.z * wi + ba.w * wr;
    let wRe2 = bb.x * wr - bb.y * wi;  let wIm2 = bb.x * wi + bb.y * wr;

    smA[top] = vec4f(ta.x + wRe0, ta.y + wIm0, ta.z + wRe1, ta.w + wIm1);
    smA[bot] = vec4f(ta.x - wRe0, ta.y - wIm0, ta.z - wRe1, ta.w - wIm1);
    smB[top] = vec4f(tb.x + wRe2, tb.y + wIm2, 0.0, 0.0);
    smB[bot] = vec4f(tb.x - wRe2, tb.y - wIm2, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn fft_rows(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id)        wid: vec3u,
) {
    let row  = wid.x;
    let tid  = lid.x;
    let base = row * N;

    let ra = bitrev7(tid);
    let rb = bitrev7(tid + 64u);
    smA[ra] = specBuf[(base + tid)       * 2u];
    smA[rb] = specBuf[(base + tid + 64u) * 2u];
    smB[ra] = specBuf[(base + tid)       * 2u + 1u];
    smB[rb] = specBuf[(base + tid + 64u) * 2u + 1u];
    workgroupBarrier();

    butterfly_stage(tid, 1u); workgroupBarrier();
    butterfly_stage(tid, 2u); workgroupBarrier();
    butterfly_stage(tid, 3u); workgroupBarrier();
    butterfly_stage(tid, 4u); workgroupBarrier();
    butterfly_stage(tid, 5u); workgroupBarrier();
    butterfly_stage(tid, 6u); workgroupBarrier();
    butterfly_stage(tid, 7u); workgroupBarrier();

    let inv = 1.0 / f32(N);
    smA[tid]       *= inv;
    smA[tid + 64u] *= inv;
    smB[tid]       *= inv;
    smB[tid + 64u] *= inv;
    workgroupBarrier();

    specBuf[(base + tid)       * 2u]      = smA[tid];
    specBuf[(base + tid)       * 2u + 1u] = smB[tid];
    specBuf[(base + tid + 64u) * 2u]      = smA[tid + 64u];
    specBuf[(base + tid + 64u) * 2u + 1u] = smB[tid + 64u];
}

@compute @workgroup_size(64)
fn fft_cols(
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(workgroup_id)        wid: vec3u,
) {
    let col = wid.x;
    let tid = lid.x;

    let ra = bitrev7(tid);
    let rb = bitrev7(tid + 64u);
    smA[ra] = specBuf[(tid * N + col)         * 2u];
    smA[rb] = specBuf[((tid + 64u) * N + col) * 2u];
    smB[ra] = specBuf[(tid * N + col)         * 2u + 1u];
    smB[rb] = specBuf[((tid + 64u) * N + col) * 2u + 1u];
    workgroupBarrier();

    butterfly_stage(tid, 1u); workgroupBarrier();
    butterfly_stage(tid, 2u); workgroupBarrier();
    butterfly_stage(tid, 3u); workgroupBarrier();
    butterfly_stage(tid, 4u); workgroupBarrier();
    butterfly_stage(tid, 5u); workgroupBarrier();
    butterfly_stage(tid, 6u); workgroupBarrier();
    butterfly_stage(tid, 7u); workgroupBarrier();

    let inv = 1.0 / f32(N);
    smA[tid]       *= inv;
    smA[tid + 64u] *= inv;
    smB[tid]       *= inv;
    smB[tid + 64u] *= inv;
    workgroupBarrier();

    specBuf[(tid * N + col)         * 2u]      = smA[tid];
    specBuf[(tid * N + col)         * 2u + 1u] = smB[tid];
    specBuf[((tid + 64u) * N + col) * 2u]      = smA[tid + 64u];
    specBuf[((tid + 64u) * N + col) * 2u + 1u] = smB[tid + 64u];
}

@compute @workgroup_size(8, 8)
fn ocean_pack(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= N || gid.y >= N) { return; }
    let idx = gid.y * N + gid.x;
    let v0  = specBuf[idx * 2u];
    let v1  = specBuf[idx * 2u + 1u];
    textureStore(oceanOut, gid.xy, vec4f(v0.x, v0.z, v1.x, 0.0));
}
