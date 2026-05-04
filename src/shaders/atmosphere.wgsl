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
    seed        : f32,
    resolution  : vec2f,
    _pad3       : vec2f,
};

struct VOut {
    @builtin(position) position : vec4f,
    @location(0) world_pos : vec3f,
};

@group(0) @binding(0) var<uniform> globals : Globals;
@group(0) @binding(1) var transmittanceLUT : texture_2d<f32>;

const R_EARTH    : f32   = 6371.0;
const R_TOP      : f32   = 6471.0;
const H_ATM      : f32   = 100.0;
const H_R        : f32   = 8.0;
const H_M        : f32   = 1.2;
const BETA_R     : vec3f = vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const BETA_M     : f32   = 0.8e-3;
const BETA_M_EXT : f32   = 0.9e-3;
const MIE_G      : f32   = 0.3;
const SUN_I      : f32   = 20.0;
const PI         : f32   = 3.14159265358979323846f;
const STEPS      : u32   = 16u;

@vertex
fn vs_main(@location(0) position: vec3f) -> VOut {
    let clip = vec4f(position.xy, position.z, 1.0);
    let wh   = globals.invViewProj * clip;
    var out: VOut;
    out.position  = clip;
    out.world_pos = wh.xyz / wh.w;
    return out;
}

fn sampleT(r: f32, mu: f32) -> vec4f {
    let u    = clamp((r - R_EARTH) / H_ATM, 0.0, 1.0);
    let v    = clamp(mu * 0.5 + 0.5, 0.0, 1.0);
    let sz   = vec2f(textureDimensions(transmittanceLUT));
    let uv_t = vec2f(u, v) * (sz - 1.0);
    let i0   = vec2u(vec2i(floor(uv_t)));
    let i1   = min(i0 + 1u, vec2u(sz) - 1u);
    let f    = fract(uv_t);
    let d00  = textureLoad(transmittanceLUT, i0, 0);
    let d10  = textureLoad(transmittanceLUT, vec2u(i1.x, i0.y), 0);
    let d01  = textureLoad(transmittanceLUT, vec2u(i0.x, i1.y), 0);
    let d11  = textureLoad(transmittanceLUT, i1, 0);
    return mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
}

fn distToAtmTop(r: f32, mu: f32) -> f32 {
    let disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    return max(0.0, -r * mu + sqrt(max(disc, 0.0)));
}

fn aces(x: vec3f) -> vec3f {
    let a = 2.51f; let b = 0.03f; let c = 2.43f; let d = 0.59f; let e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

fn phaseR(cosTheta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

fn phaseM(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (3.0 / (8.0 * PI)) * (1.0 - g2) * (1.0 + cosTheta * cosTheta)
         / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

fn skyScatter(integration_dir: vec3f, scatter_dir: vec3f, sun_dir: vec3f) -> vec3f {
    let mu_view   = max(integration_dir.y, 0.0001);
    let cos_theta = dot(scatter_dir, sun_dir);
    let pr        = phaseR(cos_theta);
    let pm        = phaseM(cos_theta, MIE_G);

    let r0    = R_EARTH;
    let d_top = min(distToAtmTop(r0, mu_view), 180.0);
    let ds    = d_top / f32(STEPS);

    var L      : vec3f = vec3f(0.0);
    var T_view : vec3f = vec3f(1.0);

    for (var i = 0u; i < STEPS; i++) {
        let t   = (f32(i) + 0.5) * ds;
        let r_i = sqrt(r0 * r0 + t * t + 2.0 * r0 * mu_view * t);
        let h_i = max(r_i - R_EARTH, 0.0);

        let mu_s = (r0 * sun_dir.y + t * cos_theta) / r_i;

        let Ts    = sampleT(r_i, mu_s);
        let Ts_r  = Ts.rgb;
        let Ts_m  = Ts.a;

        let d_r = exp(-h_i / H_R);
        let d_m = exp(-h_i / H_M);

        L += T_view * ds * (BETA_R * d_r * pr * Ts_r
                          + vec3f(BETA_M * d_m * pm * Ts_m));

        T_view *= exp(-(BETA_R * d_r + vec3f(BETA_M_EXT * d_m)) * ds);
    }

    return L * SUN_I;
}

fn hash(p: vec2f) -> f32 {
    let q = fract(p * 0.3183099 + vec2f(0.1, 0.1));
    return fract((q.x + q.y) * (q.x + q.y + 17.0));
}

fn hash3(p: vec3f) -> f32 {
    var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 19.19);
    return fract((q.x + q.y) * q.z);
}

fn noise2(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2f(1.0, 0.0)), u.x),
               mix(hash(i + vec2f(0.0, 1.0)), hash(i + vec2f(1.0, 1.0)), u.x), u.y);
}

fn noise3(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash3(i), hash3(i + vec3f(1,0,0)), u.x),
                   mix(hash3(i + vec3f(0,1,0)), hash3(i + vec3f(1,1,0)), u.x), u.y),
               mix(mix(hash3(i + vec3f(0,0,1)), hash3(i + vec3f(1,0,1)), u.x),
                   mix(hash3(i + vec3f(0,1,1)), hash3(i + vec3f(1,1,1)), u.x), u.y), u.z);
}

const CLOUD_BASE    : f32 = 2200.0;
const CLOUD_TOP     : f32 = 4500.0;
const CLOUD_STEPS   : u32 = 32u;
const CLOUD_SIGMA_E : f32 = 0.0030;
const CLOUD_SIGMA_S : f32 = 0.0025;

fn cloudSampleDensity(wp: vec3f) -> f32 {
    let h    = clamp((wp.y - CLOUD_BASE) / (CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    let prof = smoothstep(0.0, 0.12, h) * smoothstep(1.0, 0.75, h);
    if (prof < 0.001) { return 0.0; }

    let drift = vec3f(globals.timeOfDay * 80.0, 0.0, globals.timeOfDay * 40.0);
    let p     = (wp + drift) * 0.000260 + vec3f(globals.seed * 0.07, 0.0, globals.seed * 0.05);
    let n = noise3(p)              * 0.52
          + noise3(p * 2.3)        * 0.26
          + noise3(p * 5.1 + 1.7)  * 0.13
          + noise3(p * 11.0 + 3.3) * 0.065;
    return smoothstep(0.64, 0.78, n) * prof;
}

fn hg(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

struct CloudResult { color: vec3f, transmittance: f32 }

fn marchClouds(ray_o: vec3f, ray_d: vec3f, sun_dir: vec3f, sky_col: vec3f) -> CloudResult {
    var result: CloudResult;
    result.transmittance = 1.0;
    result.color = vec3f(0.0);

    var t0 = (CLOUD_BASE - ray_o.y) / ray_d.y;
    var t1 = (CLOUD_TOP  - ray_o.y) / ray_d.y;
    if (t0 > t1) { let tmp = t0; t0 = t1; t1 = tmp; }
    t0 = max(t0, 0.0);
    if (t1 <= t0) { result.transmittance = 1.0; return result; }
    t1 = min(t1, t0 + 5000.0);

    let ds  = (t1 - t0) / f32(CLOUD_STEPS);
    let cos_sun = dot(ray_d, sun_dir);
    let phase = hg(cos_sun, 0.65) * 0.6 + hg(cos_sun, -0.3) * 0.4;

    let sun_elev  = sun_dir.y;
    let day_fac   = smoothstep(-0.05, 0.15, sun_elev);
    let sun_col   = mix(vec3f(1.8, 0.8, 0.4), vec3f(1.5, 1.45, 1.40), smoothstep(0.0, 0.2, sun_elev))
                  * day_fac * 2.2;

    let jitter = hash3(ray_d * 937.31 + ray_o * 0.00017) * ds;

    var T = 1.0;

    for (var i = 0u; i < CLOUD_STEPS; i++) {
        let t  = t0 + jitter + f32(i) * ds;
        let wp = ray_o + ray_d * t;
        let density = cloudSampleDensity(wp);
        if (density < 0.001) { continue; }

        let sigma_e = CLOUD_SIGMA_E * density;
        let sigma_s = CLOUD_SIGMA_S * density;
        let dT = exp(-sigma_e * ds);

        var T_sun = 0.0;
        for (var j = 0u; j < 4u; j++) {
            let st = f32(j + 1u) * 300.0;
            T_sun += cloudSampleDensity(wp + sun_dir * st) * CLOUD_SIGMA_E * 300.0;
        }
        let beer   = exp(-T_sun);
        let powder = 1.0 - exp(-T_sun * 2.0);
        let L_sun  = sun_col * beer * powder * phase * sigma_s;

        let L_amb  = mix(vec3f(0.02, 0.02, 0.04), vec3f(1.00, 1.02, 1.05), day_fac)
                   * 0.80 * sigma_s;

        result.color += T * (L_sun + L_amb) * (1.0 - dT) / max(sigma_e, 1e-5);
        T *= dT;
        if (T < 0.01) { break; }
    }

    result.transmittance = T;
    return result;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    let view_dir  = normalize(in.world_pos - globals.cameraPos);
    let sun_dir   = globals.sunDir;
    let sun_elev  = sun_dir.y;
    let day_fac   = smoothstep(-0.05, 0.15, sun_elev);
    let tod       = globals.timeOfDay;

    var sky_dir = normalize(vec3f(view_dir.x, max(view_dir.y, 0.0), view_dir.z));

    var sky = skyScatter(sky_dir, sky_dir, sun_dir);

    let ms_ambient = vec3f(0.10, 0.28, 1.0) * 0.035 * day_fac;
    sky += ms_ambient;

    sky.g *= 1.0 - day_fac * 0.40;

    let night_sky = vec3f(0.004, 0.005, 0.018);
    sky = mix(night_sky, sky, clamp(day_fac * 1.1, 0.0, 1.0));

    let cos_sun  = dot(view_dir, sun_dir);
    let sun_disk = smoothstep(0.9997, 0.9999, cos_sun) * day_fac * 8.0;
    sky += vec3f(sun_disk) * vec3f(1.5, 1.25, 1.0);

    let night_fac    = 1.0 - day_fac;
    let moon_ang     = (fract(tod + 0.5) - 0.5) * PI;
    let moon_dir     = normalize(vec3f(-sin(moon_ang), sin(moon_ang * 2.0) * 0.8, cos(moon_ang)));
    let cos_moon     = dot(view_dir, moon_dir);
    let moon_disk    = smoothstep(0.9975, 0.9985, cos_moon) * night_fac * 0.35;
    let moon_glow    = smoothstep(0.990, 0.9975, cos_moon)  * night_fac * 0.08;
    sky += vec3f(moon_disk + moon_glow);

    let cam_above_clouds = globals.cameraPos.y > CLOUD_TOP;
    let enters_layer = select(view_dir.y > 0.01, view_dir.y < -0.01, cam_above_clouds)
                    || (globals.cameraPos.y >= CLOUD_BASE && globals.cameraPos.y <= CLOUD_TOP);
    if (enters_layer) {
        let clouds = marchClouds(globals.cameraPos, view_dir, sun_dir, sky);
        sky = sky * clouds.transmittance + clouds.color;
    }

    sky = aces(sky);
    sky = pow(clamp(sky, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

    return vec4f(sky, 1.0);
}
