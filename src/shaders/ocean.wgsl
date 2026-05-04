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
    resolution  : vec2f,
    _pad3       : vec2f,
};

@group(0) @binding(0) var<uniform> globals  : Globals;
@group(0) @binding(1) var          oceanTex : texture_2d<f32>;

const OCEAN_N    : u32 = 128u;
const MESH_N     : u32 = 256u;
const TILE_WORLD : f32 = 800.0;
const WORLD_EXT  : f32 = 40000.0;
const WORLD_HALF : f32 = 20000.0;
const WAVE_SCALE : f32 = 1.0;

struct VOut {
    @builtin(position) clip_pos  : vec4f,
    @location(0)       world_pos : vec3f,
    @location(1)       raw_uv    : vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    let quad_idx = vi / 6u;
    let local_vi = vi % 6u;
    let quad_col = quad_idx % MESH_N;
    let quad_row = quad_idx / MESH_N;

    let dc = select(0u, 1u, local_vi == 1u || local_vi == 3u || local_vi == 4u);
    let dr = select(0u, 1u, local_vi == 2u || local_vi == 4u || local_vi == 5u);

    let vc = quad_col + dc;
    let vr = quad_row + dr;

    let wx = f32(vc) / f32(MESH_N) * WORLD_EXT - WORLD_HALF;
    let wz = f32(vr) / f32(MESH_N) * WORLD_EXT - WORLD_HALF;

    let d        = sampleOcean(wx, wz);
    let cam_dist  = length(vec2f(wx - globals.cameraPos.x, wz - globals.cameraPos.z));
    let wave_fade = 1.0 - smoothstep(5000.0, 9000.0, cam_dist);
    let height   = d.r * WAVE_SCALE * wave_fade;
    let wy       = globals.seaLevel + height;

    var out: VOut;
    out.clip_pos  = globals.viewProj * vec4f(wx, wy, wz, 1.0);
    out.world_pos = vec3f(wx, wy, wz);
    out.raw_uv    = vec2f(wx, wz);
    return out;
}

// Manual bilinear blend since rgba32float can't use textureSample without float32-filterable
fn sampleOcean(wx: f32, wz: f32) -> vec4f {
    let N  = i32(OCEAN_N);
    let u  = wx / TILE_WORLD * f32(OCEAN_N);
    let v  = wz / TILE_WORLD * f32(OCEAN_N);
    let i0 = i32(floor(u));
    let j0 = i32(floor(v));
    let fx = u - f32(i0);
    let fz = v - f32(j0);

    let tx0 = u32(((i0     % N) + N) % N);
    let tx1 = u32((((i0+1) % N) + N) % N);
    let tz0 = u32(((j0     % N) + N) % N);
    let tz1 = u32((((j0+1) % N) + N) % N);

    let d00 = textureLoad(oceanTex, vec2u(tx0, tz0), 0);
    let d10 = textureLoad(oceanTex, vec2u(tx1, tz0), 0);
    let d01 = textureLoad(oceanTex, vec2u(tx0, tz1), 0);
    let d11 = textureLoad(oceanTex, vec2u(tx1, tz1), 0);

    return mix(mix(d00, d10, fx), mix(d01, d11, fx), fz);
}

fn skyColor(dir: vec3f, sunDir: vec3f) -> vec3f {
    let zenith       = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sunElevation = clamp(sunDir.y, -0.1, 1.0);
    let nightSky     = vec3f(0.01, 0.01, 0.05);
    let dawnSky      = vec3f(0.70, 0.35, 0.15);
    let daySkyTop    = vec3f(0.08, 0.25, 0.72);
    let daySkyHz     = vec3f(0.40, 0.50, 0.88);
    let dayMix       = smoothstep(-0.05, 0.15, sunElevation);
    let dawnMix      = smoothstep(-0.15, 0.0, sunElevation)
                     * (1.0 - smoothstep(0.1, 0.3, sunElevation));

    let sunDirFac    = 0.15 + 0.85 * smoothstep(-0.6, 0.4, -dot(dir, sunDir));

    var sky           = mix(nightSky, mix(daySkyHz, daySkyTop, zenith), dayMix);
    sky               = mix(sky, dawnSky, dawnMix * (1.0 - zenith) * 0.6 * sunDirFac);
    return sky;
}

fn aces(x: vec3f) -> vec3f {
    let a = 2.51f; let b = 0.03f; let c = 2.43f; let d = 0.59f; let e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    let d = sampleOcean(in.raw_uv.x, in.raw_uv.y);

    // N = (-dH/dx, 1, -dH/dz) from gradient texture
    let slope_x = d.g;
    let slope_z = d.b;
    let N_vec   = normalize(vec3f(-slope_x, 1.0, -slope_z));

    let V      = normalize(globals.cameraPos - in.world_pos);
    let sunDir = globals.sunDir;

    let sunElevation = sunDir.y;
    let dayFactor    = smoothstep(-0.05, 0.15, sunElevation);

    // Fresnel (Schlick)
    let NdotV   = max(dot(N_vec, V), 0.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - NdotV, 5.0);

    let reflDir = reflect(-V, N_vec);
    let skyRefl = skyColor(reflDir, sunDir) * fresnel;

    let deepBlue = vec3f(0.02, 0.09, 0.20);
    let ambient  = deepBlue * (0.12 + 0.18 * dayFactor) * (1.0 - fresnel);

    let sunColorDay  = vec3f(1.4, 1.2, 0.9);
    let sunColorDawn = vec3f(1.5, 0.6, 0.2);
    let sunColor     = mix(sunColorDawn, sunColorDay,
                           smoothstep(0.0, 0.2, sunElevation)) * dayFactor;
    let refl_sun = reflect(-sunDir, N_vec);
    let spec     = pow(max(dot(refl_sun, V), 0.0), 256.0) * dayFactor * 3.0;

    var color = ambient + skyRefl + spec * sunColor;

    let dist      = length(in.world_pos - globals.cameraPos);
    let avgH      = (globals.cameraPos.y + in.world_pos.y) * 0.5;
    let hScale    = 1.0 + 2.5 * exp(-max(avgH, 0.0) * 0.004);
    let fogFac    = 1.0 - exp(-dist * 0.00005 * hScale);
    let rd        = normalize(in.world_pos - globals.cameraPos);
    let sunAmt    = max(dot(rd, sunDir), 0.0);
    let fogBase   = skyColor(rd, sunDir);
    let fogColor  = mix(fogBase, vec3f(1.0, 0.9, 0.7), pow(sunAmt, 8.0) * dayFactor);
    color         = mix(color, fogColor, clamp(fogFac, 0.0, 1.0));

    color = aces(color);
    color = pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
    let screenUV = in.clip_pos.xy / globals.resolution;
    let vig = 0.5 + 0.5 * pow(16.0 * screenUV.x * screenUV.y * (1.0 - screenUV.x) * (1.0 - screenUV.y), 0.15);
    color *= vig;
    let grey = dot(color, vec3f(0.299, 0.587, 0.114));
    color = mix(color, vec3f(grey), 0.12);
    color = pow(color, vec3f(0.97, 1.0, 1.03));

    return vec4f(color, 1.0);
}
