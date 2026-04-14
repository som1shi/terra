// Tessendorf FFT ocean — mesh grid displaced by height texture
// Texture layout (rgba32float): r=height, g=∂H/∂x, b=∂H/∂z, a=unused

struct Globals {
    viewProj    : mat4x4f,
    invViewProj : mat4x4f,
    sunDir      : vec3f,
    _pad0       : f32,
    cameraPos   : vec3f,
    _pad1       : f32,
    time        : f32,
    timeOfDay   : f32,
    seaLevel    : f32,
    _pad2       : f32,
};

@group(0) @binding(0) var<uniform> globals  : Globals;
@group(0) @binding(1) var          oceanTex : texture_2d<f32>;

const OCEAN_N    : u32 = 128u;    // FFT texture resolution
const MESH_N     : u32 = 256u;    // rendering grid resolution
const TILE_WORLD : f32 = 800.0;   // world size of one FFT tile (for UV tiling)
const WORLD_EXT  : f32 = 16000.0; // ocean extends well beyond the 4096-unit island
const WORLD_HALF : f32 = 8000.0;
const WAVE_SCALE : f32 = 1.0;     // additional height multiplier (tune for look)

// ── Vertex shader ─────────────────────────────────────────────────────────────
struct VOut {
    @builtin(position) clip_pos  : vec4f,
    @location(0)       world_pos : vec3f,
    @location(1)       raw_uv    : vec2f,  // raw texel coords (0..N), interpolated
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    let quad_idx = vi / 6u;
    let local_vi = vi % 6u;
    let quad_col = quad_idx % MESH_N;
    let quad_row = quad_idx / MESH_N;

    // Two-triangle quad (CCW):
    //   tri0: A(0,0) B(1,0) C(0,1)
    //   tri1: B(1,0) D(1,1) C(0,1)
    let dc = select(0u, 1u, local_vi == 1u || local_vi == 3u || local_vi == 4u);
    let dr = select(0u, 1u, local_vi == 2u || local_vi == 4u || local_vi == 5u);

    let vc = quad_col + dc;
    let vr = quad_row + dr;

    // World position spread across terrain extent
    let wx = f32(vc) / f32(MESH_N) * WORLD_EXT - WORLD_HALF;
    let wz = f32(vr) / f32(MESH_N) * WORLD_EXT - WORLD_HALF;

    // Tile the FFT texture over the world (with correct negative-modulo wrapping)
    let ix = i32(round(wx / TILE_WORLD * f32(OCEAN_N)));
    let iz = i32(round(wz / TILE_WORLD * f32(OCEAN_N)));
    let tx = u32(((ix % i32(OCEAN_N)) + i32(OCEAN_N)) % i32(OCEAN_N));
    let tz = u32(((iz % i32(OCEAN_N)) + i32(OCEAN_N)) % i32(OCEAN_N));

    let d      = textureLoad(oceanTex, vec2u(tx, tz), 0);
    let height = d.r * WAVE_SCALE;
    let wy     = globals.seaLevel + height;

    var out: VOut;
    out.clip_pos  = globals.viewProj * vec4f(wx, wy, wz, 1.0);
    out.world_pos = vec3f(wx, wy, wz);
    out.raw_uv    = vec2f(wx, wz);   // world xz for FS tiling lookup
    return out;
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn skyColor(dir: vec3f, sunDir: vec3f) -> vec3f {
    let zenith       = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sunElevation = clamp(sunDir.y, -0.1, 1.0);
    let nightSky     = vec3f(0.01, 0.01, 0.05);
    let dawnSky      = vec3f(0.70, 0.35, 0.15);
    let daySkyTop    = vec3f(0.08, 0.25, 0.72);
    let daySkyHz     = vec3f(0.40, 0.65, 0.90);
    let dayMix       = smoothstep(-0.05, 0.15, sunElevation);
    let dawnMix      = smoothstep(-0.15, 0.0, sunElevation)
                     * (1.0 - smoothstep(0.1, 0.3, sunElevation));
    var sky           = mix(nightSky, mix(daySkyHz, daySkyTop, zenith), dayMix);
    sky               = mix(sky, dawnSky, dawnMix * (1.0 - zenith) * 0.6);
    return sky;
}

fn aces(x: vec3f) -> vec3f {
    let a = 2.51f; let b = 0.03f; let c = 2.43f; let d = 0.59f; let e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

// ── Fragment shader ───────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    // Tiled slope lookup matching VS
    let ix = i32(round(in.raw_uv.x / TILE_WORLD * f32(OCEAN_N)));
    let iz = i32(round(in.raw_uv.y / TILE_WORLD * f32(OCEAN_N)));
    let tx = u32(((ix % i32(OCEAN_N)) + i32(OCEAN_N)) % i32(OCEAN_N));
    let tz = u32(((iz % i32(OCEAN_N)) + i32(OCEAN_N)) % i32(OCEAN_N));
    let d  = textureLoad(oceanTex, vec2u(tx, tz), 0);

    // Build normal from gradient: surface y=H(x,z) → N = (-∂H/∂x, 1, -∂H/∂z)
    let slope_x = d.g;
    let slope_z = d.b;
    let N_vec   = normalize(vec3f(-slope_x, 1.0, -slope_z));

    let V      = normalize(globals.cameraPos - in.world_pos);
    let sunDir = globals.sunDir;

    let sunElevation = sunDir.y;
    let dayFactor    = smoothstep(-0.05, 0.15, sunElevation);

    // Fresnel (Schlick)
    let NdotV  = max(dot(N_vec, V), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);

    // Sky reflection
    let reflDir = reflect(-V, N_vec);
    let skyRefl = skyColor(reflDir, sunDir) * fresnel;

    // Deep-water diffuse
    let deepBlue = vec3f(0.02, 0.09, 0.20);
    let ambient  = deepBlue * (0.12 + 0.18 * dayFactor) * (1.0 - fresnel);

    // Sun specular
    let sunColorDay  = vec3f(1.4, 1.2, 0.9);
    let sunColorDawn = vec3f(1.5, 0.6, 0.2);
    let sunColor     = mix(sunColorDawn, sunColorDay,
                           smoothstep(0.0, 0.2, sunElevation)) * dayFactor;
    let refl_sun = reflect(-sunDir, N_vec);
    let spec     = pow(max(dot(refl_sun, V), 0.0), 256.0) * dayFactor * 3.0;

    var color = ambient + skyRefl + spec * sunColor;

    // Distance fog
    let dist     = length(in.world_pos - globals.cameraPos);
    let fogFac   = 1.0 - exp(-dist * 0.00005);
    let fogColor = skyColor(normalize(in.world_pos - globals.cameraPos), sunDir);
    color        = mix(color, fogColor, clamp(fogFac, 0.0, 1.0));

    // Tonemap + gamma
    color = aces(color);
    color = pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

    return vec4f(color, 1.0);
}
