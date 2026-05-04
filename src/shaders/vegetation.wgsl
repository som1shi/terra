const SIZE         : i32 = 512;
const MAX_INSTANCES: u32 = 50000u;
const HEIGHT_SCALE : f32 = 600.0;
const WORLD_SCALE  : f32 = 4096.0;
const WORLD_HALF   : f32 = 2048.0;

struct DrawArgs {
    indexCount    : u32,
    instanceCount : atomic<u32>,
    firstIndex    : u32,
    baseVertex    : u32,
    firstInstance : u32,
};

@group(0) @binding(0) var heightTex      : texture_2d<f32>;
@group(0) @binding(1) var normalTex      : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> drawArgs  : DrawArgs;
@group(0) @binding(3) var<storage, read_write> positions : array<vec4f>;
@group(0) @binding(4) var smoothAccumTex : texture_2d<f32>;

fn hash(x: u32, y: u32, salt: u32) -> f32 {
    var h = (x * 2747636419u) ^ (y * 2246822519u) ^ salt;
    h ^= h >> 16u;
    h *= 0x85ebca6bu;
    h ^= h >> 13u;
    return f32(h & 0xFFFFu) / 65535.0;
}

fn hash2D(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453);
}

@compute @workgroup_size(8, 8)
fn place_vegetation(@builtin(global_invocation_id) id: vec3u) {
    let coord = vec2i(id.xy);
    if (coord.x >= SIZE || coord.y >= SIZE) { return; }

    let h = textureLoad(heightTex, coord, 0).r;
    let N = normalize(textureLoad(normalTex, coord, 0).xyz * 2.0 - 1.0);

    if (h < 0.12 || h > 0.28 || N.y < 0.75) { return; }

    var maxFlow = textureLoad(smoothAccumTex, coord, 0).r;
    let cardinals = array<vec2i, 4>(vec2i(-1,0), vec2i(1,0), vec2i(0,-1), vec2i(0,1));
    for (var i = 0; i < 4; i++) {
        let nc = clamp(coord + cardinals[i], vec2i(0), vec2i(SIZE - 1));
        maxFlow = max(maxFlow, textureLoad(smoothAccumTex, nc, 0).r);
    }
    if (log(max(maxFlow, 1.0)) > 2.2) { return; }

    let clusterHash = hash2D(floor(vec2f(f32(coord.x), f32(coord.y)) / 32.0));
    let threshold   = 0.03 * clusterHash;
    if (hash(id.x, id.y, 0u) > threshold) { return; }

    let idx = atomicAdd(&drawArgs.instanceCount, 1u);
    if (idx >= MAX_INSTANCES) { return; }

    let uv   = vec2f(f32(coord.x), f32(coord.y)) / f32(SIZE - 1);
    let wx   = uv.x * WORLD_SCALE - WORLD_HALF;
    let wz   = uv.y * WORLD_SCALE - WORLD_HALF;
    let wy   = h * HEIGHT_SCALE;
    let seed = hash2D(vec2f(f32(coord.x) * 13.7, f32(coord.y) * 7.3));

    positions[idx] = vec4f(wx, wy, wz, seed);
}
