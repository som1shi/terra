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

@group(0) @binding(0) var<uniform> globals : Globals;
@group(0) @binding(1) var heightmapTex     : texture_2d<f32>;
@group(0) @binding(2) var heightmapSampler : sampler;
@group(0) @binding(3) var normalTex        : texture_2d<f32>;
@group(0) @binding(4) var normalSampler    : sampler;
@group(0) @binding(5) var aoTex            : texture_2d<f32>;
@group(0) @binding(6) var aoSampler        : sampler;
@group(0) @binding(7) var accumTex         : texture_2d<f32>;

const HEIGHT_SCALE : f32 = 600.0;
const WORLD_HALF   : f32 = 2048.0;
const TEXEL_SIZE   : f32 = 1.0 / 512.0;

struct VertexInput {
    @location(0) xz : vec2f,
    @location(1) uv : vec2f,
};

struct VertexOutput {
    @builtin(position) clip_pos   : vec4f,
    @location(0)       world_pos  : vec3f,
    @location(1)       uv         : vec2f,
    @location(2)       fog_factor : f32,
};

fn sampleHeight(uv: vec2f) -> f32 {
    return textureLoad(heightmapTex, vec2i(uv * 512.0), 0).r * HEIGHT_SCALE;
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let uv = in.uv;
    let h  = sampleHeight(uv);

    // Two-scale noise on snap threshold produces organic coastline that doesn't follow the grid
    let snapNoise  = (multiNoise(in.xz * 0.003) - 0.5) * 30.0
                   + (valueNoise(in.xz * 0.01)  - 0.5) * 14.0;
    let snapThresh = globals.seaLevel + 30.0 + snapNoise;

    let onEdge     = uv.x <= TEXEL_SIZE * 2.0 || uv.x >= 1.0 - TEXEL_SIZE * 2.0 ||
                     uv.y <= TEXEL_SIZE * 2.0 || uv.y >= 1.0 - TEXEL_SIZE * 2.0;
    let edgeFactor = select(0.0, 1.0, onEdge);
    let snapFactor = max(edgeFactor, 1.0 - smoothstep(snapThresh - 3.0, snapThresh + 3.0, h));
    let y          = mix(h, globals.seaLevel - 30.0, snapFactor);
    let world_pos  = vec3f(in.xz.x, y, in.xz.y);

    out.clip_pos  = globals.viewProj * vec4f(world_pos, 1.0);
    out.world_pos = world_pos;
    out.uv        = uv;

    let dist        = length(world_pos - globals.cameraPos);
    let avgHeight   = (globals.cameraPos.y + world_pos.y) * 0.5;
    let heightScale = 1.0 + 2.5 * exp(-max(avgHeight, 0.0) * 0.004);
    out.fog_factor  = 1.0 - exp(-dist * 0.00005 * heightScale);

    return out;
}

fn skyColor(dir: vec3f, sunDir: vec3f, tod: f32) -> vec3f {
    let zenith       = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let sunElevation = clamp(sunDir.y, -0.1, 1.0);

    let nightSky  = vec3f(0.01, 0.01, 0.05);
    let dawnSky   = vec3f(0.7, 0.35, 0.15);
    let daySkyTop = vec3f(0.08, 0.25, 0.72);
    let daySkyHz  = vec3f(0.4, 0.65, 0.9);

    let dayMix  = smoothstep(-0.05, 0.15, sunElevation);
    let dawnMix = smoothstep(-0.15, 0.0, sunElevation) * (1.0 - smoothstep(0.1, 0.3, sunElevation));

    var sky = mix(nightSky, mix(daySkyHz, daySkyTop, zenith), dayMix);
    sky = mix(sky, dawnSky, dawnMix * (1.0 - zenith) * 0.6);
    return sky;
}

fn aces(x: vec3f) -> vec3f {
    let a = 2.51f; let b = 0.03f; let c = 2.43f; let d = 0.59f; let e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

fn hash2f(p: vec2f) -> f32 {
    var q = fract(p / 289.0) * 289.0;
    q = vec2f(dot(q, vec2f(127.1, 311.7)), dot(q, vec2f(269.5, 183.3)));
    return fract(sin(dot(q, vec2f(1.0, 1.0))) * 43758.5453);
}

fn valueNoise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash2f(i + vec2f(0.0, 0.0)), hash2f(i + vec2f(1.0, 0.0)), u.x),
        mix(hash2f(i + vec2f(0.0, 1.0)), hash2f(i + vec2f(1.0, 1.0)), u.x),
        u.y
    );
}

fn multiNoise(p: vec2f) -> f32 {
    return valueNoise(p)       * 0.5
         + valueNoise(p * 2.1) * 0.3
         + valueNoise(p * 4.3) * 0.2;
}

fn sampleHeightBilinear(uv: vec2f) -> f32 {
    let pix = uv * 512.0 - 0.5;
    let pi  = vec2i(i32(floor(pix.x)), i32(floor(pix.y)));
    let pf  = fract(pix);
    let h00 = textureLoad(heightmapTex, clamp(pi + vec2i(0, 0), vec2i(0), vec2i(511)), 0).r;
    let h10 = textureLoad(heightmapTex, clamp(pi + vec2i(1, 0), vec2i(0), vec2i(511)), 0).r;
    let h01 = textureLoad(heightmapTex, clamp(pi + vec2i(0, 1), vec2i(0), vec2i(511)), 0).r;
    let h11 = textureLoad(heightmapTex, clamp(pi + vec2i(1, 1), vec2i(0), vec2i(511)), 0).r;
    return mix(mix(h00, h10, pf.x), mix(h01, h11, pf.x), pf.y) * HEIGHT_SCALE;
}

fn getSoftShadow(worldPos: vec3f, lightDir: vec3f) -> f32 {
    if (lightDir.y <= 0.0) { return 0.2; }

    const softness : f32 = 8.0;
    const STEP_MIN : f32 = 8.0;
    const STEP_MAX : f32 = 60.0;

    let bias_dist    = clamp(8.0 / max(lightDir.y, 0.04), 8.0, 120.0);
    let origin       = worldPos + vec3f(0.0, bias_dist, 0.0);

    var res           : f32 = 1.0;
    var step_distance : f32 = STEP_MIN;

    for (var i = 0; i < 50; i++) {
        let rayPos        = origin + lightDir * step_distance;
        let uv            = (vec2f(rayPos.x, rayPos.z) + WORLD_HALF) / (WORLD_HALF * 2.0);
        if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { break; }

        let terrainHeight = sampleHeightBilinear(uv);
        let dist          = rayPos.y - terrainHeight;

        if (dist < 0.0) { return 0.2; }

        res           = min(res, softness * dist / step_distance);
        step_distance += clamp(dist, STEP_MIN, STEP_MAX);
    }

    return clamp(res, 0.2, 1.0);
}

fn getTriplanarNoise(worldPos: vec3f, normal: vec3f, scale: f32) -> f32 {
    let w    = pow(abs(normal), vec3f(4.0));
    let wsum = max(w.x + w.y + w.z, 0.001);
    let nXZ  = multiNoise(worldPos.xz * scale);
    let nXY  = multiNoise(worldPos.xy * scale);
    let nYZ  = multiNoise(worldPos.yz * scale);
    return (nXZ * w.y + nXY * w.z + nYZ * w.x) / wsum;
}

fn getTriplanarColor(worldPos: vec3f, normal: vec3f, colorA: vec3f, colorB: vec3f, scale: f32) -> vec3f {
    let w    = pow(abs(normal), vec3f(6.0));
    let wsum = max(w.x + w.y + w.z, 0.001);
    let nXZ  = multiNoise(worldPos.xz * scale);
    let nXY  = multiNoise(worldPos.xy * scale);
    let nYZ  = multiNoise(worldPos.yz * scale);
    let n    = (nXZ * w.y + nXY * w.z + nYZ * w.x) / wsum;
    return mix(colorA, colorB, n);
}

fn getSmoothFlow(texCoords: vec2f, texSize: vec2f) -> f32 {
    let unormCoords = texCoords * texSize - 0.5;
    let iCoords     = vec2i(floor(unormCoords));
    let fCoords     = fract(unormCoords);
    let sz          = vec2i(texSize) - 1;

    let tl = textureLoad(accumTex, clamp(iCoords,               vec2i(0), sz), 0).r;
    let tr = textureLoad(accumTex, clamp(iCoords + vec2i(1, 0), vec2i(0), sz), 0).r;
    let bl = textureLoad(accumTex, clamp(iCoords + vec2i(0, 1), vec2i(0), sz), 0).r;
    let br = textureLoad(accumTex, clamp(iCoords + vec2i(1, 1), vec2i(0), sz), 0).r;

    return mix(mix(tl, tr, fCoords.x), mix(bl, br, fCoords.x), fCoords.y);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let sunDir = globals.sunDir;
    let tod    = globals.timeOfDay;
    let uv     = in.uv;

    let N = normalize(textureSampleLevel(normalTex, normalSampler, uv, 0.0).xyz * 2.0 - 1.0);

    const BR : f32 = TEXEL_SIZE * 8.0;
    let N_b = normalize(
        textureSampleLevel(normalTex, normalSampler, uv,                     0.0).xyz * 2.0 - 1.0 +
        textureSampleLevel(normalTex, normalSampler, uv + vec2f( BR,  0.0),  0.0).xyz * 2.0 - 1.0 +
        textureSampleLevel(normalTex, normalSampler, uv + vec2f(-BR,  0.0),  0.0).xyz * 2.0 - 1.0 +
        textureSampleLevel(normalTex, normalSampler, uv + vec2f( 0.0,  BR),  0.0).xyz * 2.0 - 1.0 +
        textureSampleLevel(normalTex, normalSampler, uv + vec2f( 0.0, -BR),  0.0).xyz * 2.0 - 1.0
    );

    let h_smooth = sampleHeightBilinear(uv);

    let sunElevation = sunDir.y;
    let dayFactor    = smoothstep(-0.05, 0.15, sunElevation);

    let NdotL     = max(dot(N, sunDir), 0.0);
    let rawShadow = getSoftShadow(in.world_pos, sunDir);

    let sun_term = NdotL * vec3f(1.64, 1.27, 0.99)
                 * pow(vec3f(rawShadow), vec3f(1.0, 1.2, 1.5)) * dayFactor;

    let sky_term = (0.5 + 0.5 * N.y) * vec3f(0.16, 0.20, 0.28) * dayFactor
                 + (0.5 + 0.5 * N.y) * vec3f(0.006, 0.006, 0.018) * (1.0 - dayFactor);

    let ind_dir  = normalize(sunDir * vec3f(-1.0, 0.0, -1.0));
    let ind_term = max(dot(N, ind_dir), 0.0) * vec3f(0.40, 0.28, 0.20) * dayFactor;

    let hbao        = textureSampleLevel(aoTex, aoSampler, uv, 0.0).r;
    let hbao_factor = mix(0.35, 1.0, hbao);

    let altitude = clamp(h_smooth / HEIGHT_SCALE, 0.0, 1.0);
    let slope    = 1.0 - N_b.y;
    let seaLine  = globals.seaLevel / HEIGHT_SCALE;

    let triGrass = getTriplanarColor(in.world_pos, N_b,
                       vec3f(0.15, 0.25, 0.08), vec3f(0.22, 0.35, 0.16), 1.0 / 55.0);
    let triRock  = getTriplanarColor(in.world_pos, N_b,
                       vec3f(0.12, 0.10, 0.08), vec3f(0.28, 0.22, 0.16), 1.0 / 42.0);
    let triDirt  = getTriplanarColor(in.world_pos, N_b,
                       vec3f(0.22, 0.16, 0.10), vec3f(0.30, 0.22, 0.14), 1.0 / 35.0);

    let jw     = pow(abs(N_b), vec3f(4.0));
    let jwt    = max(jw.x + jw.y + jw.z, 0.001);
    let jxz    = multiNoise(in.world_pos.xz * 0.05);
    let jxy    = multiNoise(in.world_pos.xy * 0.05);
    let jyz    = multiNoise(in.world_pos.yz * 0.05);
    let jitter = ((jxz * jw.y + jxy * jw.z + jyz * jw.x) / jwt - 0.5) * 0.12;

    let gNoise = getTriplanarNoise(in.world_pos, N_b, 1.0 / 40.0);
    let dNoise = getTriplanarNoise(in.world_pos, N_b, 1.0 / 28.0);
    let rNoise = getTriplanarNoise(in.world_pos, N_b, 1.0 / 32.0);

    let deepWater = vec3f(0.05, 0.12, 0.18);
    let sand      = vec3f(0.45, 0.40, 0.30);
    let snow      = vec3f(0.92, 0.95, 1.00);

    let grassColor = triGrass * mix(0.80, 1.15, gNoise);
    let dirtColor  = triDirt  * mix(0.85, 1.10, dNoise);
    let rockColor  = triRock  * mix(0.80, 1.20, rNoise);

    // Per-pixel coast from bilinear heightmap eliminates vertex-grid staircase edges
    let coastNoise  = (multiNoise(in.world_pos.xz * 0.003) - 0.5) * 30.0
                    + (valueNoise(in.world_pos.xz * 0.01)  - 0.5) * 14.0;
    let coastThresh = globals.seaLevel + 30.0 + coastNoise;
    let coastBlend  = smoothstep(-12.0, 12.0, h_smooth - coastThresh);
    let h_rel       = mix(-30.0, h_smooth - globals.seaLevel, coastBlend);
    let shoreN      = multiNoise(in.world_pos.xz * 0.006) * 10.0 - 5.0;
    let sandNight   = mix(vec3f(0.08, 0.07, 0.06), sand, dayFactor);

    let beachNoise  = multiNoise(in.world_pos.xz * 0.002);
    let isBeach     = smoothstep(0.35, 0.55, beachNoise);
    let beachCol    = mix(deepWater, sandNight,
                          smoothstep(-12.0, 8.0, h_rel + shoreN));
    let rockyCol    = mix(deepWater, rockColor * 0.7,
                          smoothstep(-5.0, 10.0, h_rel));
    let waterSand   = mix(rockyCol, beachCol, isBeach);
    var terrain_color = mix(waterSand, grassColor,
                            smoothstep(3.0, 22.0, h_rel));
    terrain_color     = mix(terrain_color, dirtColor,
                            smoothstep(0.18 + jitter, 0.40 + jitter, altitude));
    terrain_color     = mix(terrain_color, rockColor,
                            smoothstep(0.36 + jitter, 0.62 + jitter, altitude));

    let slope_cliff = smoothstep(0.15 + jitter * 0.3, 0.42 + jitter * 0.3, slope);
    // Snap-displaced geometry: large positive (h_smooth - world_pos.y) means vertex was pulled below natural height
    let snap_cliff  = smoothstep(4.0, 16.0, h_smooth - in.world_pos.y);
    terrain_color   = mix(terrain_color, rockColor, max(slope_cliff, snap_cliff));

    let snow_alt  = smoothstep(0.52, 0.66, altitude + jitter * 0.30);
    let snow_flat = smoothstep(0.72, 0.88, N_b.y);
    terrain_color = mix(terrain_color, snow, snow_alt * snow_flat);

    // Bilinear flow sampling with 3x3 distance-weighted kernel for smooth river edges
    var flowSum  = 0.0;
    var flowMax  = 0.0;
    var flowW    = 0.0;
    for (var dx = -1; dx <= 1; dx += 1) {
        for (var dy = -1; dy <= 1; dy += 1) {
            let d2  = f32(dx * dx + dy * dy);
            let w   = 1.0 - d2 * 0.25;
            let off = vec2f(f32(dx), f32(dy)) * TEXEL_SIZE;
            let s   = getSmoothFlow(uv + off, vec2f(512.0));
            flowSum += s * w;
            flowMax  = max(flowMax, s);
            flowW   += w;
        }
    }
    let flowAvg   = flowSum / flowW;
    let logMax    = log(max(flowMax, 1.0));
    let logAvg    = log(max(flowAvg, 1.0));
    let isFlat     = smoothstep(0.88, 0.96, N_b.y);
    let slopeMask  = 1.0 - smoothstep(0.96, 0.998, N_b.y);

    let riverNoise = multiNoise(in.world_pos.xz * 0.012) * 1.6 - 0.8;
    let riverBlend = smoothstep(6.0 + riverNoise, 7.8 + riverNoise, logMax)
                   * smoothstep(4.0 + riverNoise, 6.5 + riverNoise, logAvg)
                   * isFlat * slopeMask;
    let riverColor = vec3f(0.06, 0.22, 0.55);
    terrain_color  = mix(terrain_color, riverColor, riverBlend * 0.92);

    let cliffMask  = max(snap_cliff, slope_cliff);
    let flowGate   = smoothstep(2.2, 4.5, logMax);
    let wfStrength = cliffMask * flowGate * (1.0 - isFlat);

    if (wfStrength > 0.01) {
        let time = globals.time;
        let wp   = in.world_pos;

        let wetRock = vec3f(0.05, 0.07, 0.10);
        terrain_color = mix(terrain_color, wetRock, wfStrength * 0.70);

        let hU_raw = (wp.x * 0.60 + wp.z * 0.80) * 0.022;

        let waveFreq  = 2.8;
        let waveAmp   = 0.12;
        let waveSpeed = 0.9;
        let wobble = sin(wp.y * waveFreq + time * waveSpeed) * waveAmp;
        let hU     = hU_raw + wobble;

        let scrollSpeed = 5.0;
        let vU = wp.y * 0.18 - time * scrollSpeed;

        let n0 = valueNoise(vec2f(hU * 1.0, vU * 1.0));
        let n1 = valueNoise(vec2f(hU * 2.2 + 3.7, vU * 2.2 + 1.3));
        let rawNoise = n0 * 0.70 + n1 * 0.30;

        let streamNoise = pow(rawNoise, 1.4);
        let waterBody   = smoothstep(0.28, 0.72, streamNoise);

        let edgeFoam  = smoothstep(0.62, 0.80, streamNoise);
        let foamScroll = wp.y * 0.35 - time * scrollSpeed * 1.8;
        let foamNoise  = valueNoise(vec2f(hU * 3.0 + 7.1, foamScroll));
        let intFoam    = smoothstep(0.55, 0.78, foamNoise) * waterBody;
        let foam       = clamp(edgeFoam + intFoam * 0.6, 0.0, 1.0);

        let cliffTop    = globals.seaLevel + 80.0;
        let topFade     = smoothstep(0.0, 0.35, (cliffTop - wp.y) / 120.0);
        let baseFade    = smoothstep(0.0, 0.30, (wp.y - globals.seaLevel) / 60.0);
        let vertMask    = topFade * baseFade;

        let colGate = valueNoise(vec2f(floor(hU_raw * 8.0) * 0.73, 11.5));
        let gated   = select(0.0, 1.0, colGate > 0.42);

        let waterColorDeep  = vec3f(0.10, 0.38, 0.80);
        let waterColorLight = vec3f(0.35, 0.65, 0.95);
        let vertGrad  = clamp((wp.y - globals.seaLevel) / 150.0, 0.0, 1.0);
        let bodyColor = mix(waterColorDeep, waterColorLight, vertGrad * 0.5 + rawNoise * 0.2);
        let foamColor = vec3f(0.95, 0.98, 1.00);

        var wfColor = mix(bodyColor, foamColor, foam) * mix(0.25, 1.0, dayFactor);

        let wfN    = normalize(vec3f(0.0, 0.12, 1.0));
        let wfRefl = reflect(-sunDir, wfN);
        let wfSpec = pow(max(dot(wfRefl, normalize(globals.cameraPos - wp)), 0.0), 32.0)
                     * dayFactor * 2.5 * foam;

        let wfBlend   = wfStrength * waterBody * vertMask * gated;
        terrain_color = mix(terrain_color, wfColor, wfBlend);
        terrain_color = terrain_color + wfSpec * vec3f(1.64, 1.27, 0.99) * dayFactor * wfStrength * gated * 0.5;
    }

    let t    = uv;
    let ts   = TEXEL_SIZE * 4.0;
    let h_c  = textureLoad(heightmapTex, vec2i(t * 512.0), 0).r;
    let h_px = textureLoad(heightmapTex, vec2i((t + vec2f( ts, 0.0)) * 512.0), 0).r;
    let h_nx = textureLoad(heightmapTex, vec2i((t + vec2f(-ts, 0.0)) * 512.0), 0).r;
    let h_py = textureLoad(heightmapTex, vec2i((t + vec2f(0.0,  ts)) * 512.0), 0).r;
    let h_ny = textureLoad(heightmapTex, vec2i((t + vec2f(0.0, -ts)) * 512.0), 0).r;
    let h_avg      = (h_px + h_nx + h_py + h_ny) * 0.25;
    let crevice_ao = clamp(1.0 - 2.5 * max(0.0, h_avg - h_c), 0.4, 1.0);

    var color = terrain_color * (sun_term + sky_term * hbao_factor + ind_term * hbao_factor) * crevice_ao;

    let V          = normalize(globals.cameraPos - in.world_pos);
    let riverRefl  = reflect(-sunDir, N);
    let specGate   = smoothstep(0.25, 0.65, riverBlend);
    let riverSpec  = pow(max(dot(riverRefl, V), 0.0), 96.0) * 2.8 * dayFactor * rawShadow * specGate;
    color += riverSpec * vec3f(1.64, 1.27, 0.99) * dayFactor;

    let lakeFill   = smoothstep(6.0, 8.0, logMax) * isFlat * 0.15;
    let sheenColor = vec3f(0.02, 0.18, 0.52);
    color          = color + sheenColor * (riverBlend * 0.12 + lakeFill) * dayFactor;

    // Shore blend: fade terrain near sea level toward ocean color to hide mesh-intersection stripes
    let shore_dist  = clamp(h_rel, 0.0, 1.0e9);
    let shore_blend = (1.0 - smoothstep(0.0, 35.0, shore_dist)) * mix(0.08, 0.55, dayFactor);
    color           = mix(color, vec3f(0.04, 0.10, 0.22) * max(dayFactor, 0.1), shore_blend);

    let rd        = normalize(in.world_pos - globals.cameraPos);
    let sunAmount = max(dot(rd, sunDir), 0.0);
    let baseFog   = skyColor(rd, sunDir, tod);
    let fogCol    = mix(baseFog, vec3f(1.0, 0.9, 0.7), pow(sunAmount, 8.0) * dayFactor);
    color         = mix(color, fogCol, clamp(in.fog_factor, 0.0, 1.0));

    color = aces(color);
    color = pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

    let screenUV = in.clip_pos.xy / globals.resolution;
    let vig = 0.5 + 0.5 * pow(16.0 * screenUV.x * screenUV.y
            * (1.0 - screenUV.x) * (1.0 - screenUV.y), 0.15);
    color *= vig;
    let grey = dot(color, vec3f(0.299, 0.587, 0.114));
    color = mix(color, vec3f(grey), 0.12);
    color = pow(color, vec3f(0.97, 1.0, 1.03));

    return vec4f(color, 1.0);
}
