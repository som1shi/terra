struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) world_pos : vec3f,
};

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
  fogDensity  : f32,
  _fpad0      : f32,
  _fpad1      : f32,
  _fpad2      : f32,
};

// Climate index: 0=temperate 1=arid 2=tropical 3=arctic 4=stormy
struct ClimateUniform { index: f32, _p0: f32, _p1: f32, _p2: f32, };

@group(0) @binding(0) var<uniform> globals : Globals;
@group(1) @binding(0) var<uniform> clim    : ClimateUniform;

// ── Noise ─────────────────────────────────────────────────────────────────────
fn hash(p: vec2f) -> f32 {
  let q = fract(p * 0.3183099 + vec2f(0.1, 0.1));
  return fract((q.x + q.y) * (q.x + q.y + 17.0));
}
fn noise(p: vec2f) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i+vec2f(1,0)), u.x),
             mix(hash(i+vec2f(0,1)), hash(i+vec2f(1,1)), u.x), u.y);
}
fn cloudNoise(p: vec2f) -> f32 {
  return noise(p)*0.5 + noise(p*2.0)*0.25 + noise(p*4.0)*0.125;
}

// ── Per-climate sky palette ───────────────────────────────────────────────────
struct SkyPalette {
  horizon  : vec3f,
  zenith   : vec3f,
  sunset   : vec3f,
  night    : vec3f,
  cloudCov : f32,   // 0-1 cloud coverage
};

fn getSkyPalette(ci: f32, seed: f32) -> SkyPalette {
  var p: SkyPalette;
  let hs = sin(seed * 0.01) * 0.05;
  if (ci < 0.5) {
    // Temperate
    p.horizon  = vec3f(0.60+hs, 0.70, 0.90-hs);
    p.zenith   = vec3f(0.10, 0.28, 0.80+hs);
    p.sunset   = vec3f(1.00, 0.35, 0.15);
    p.night    = vec3f(0.01, 0.01, 0.05);
    p.cloudCov = 0.40;
  } else if (ci < 1.5) {
    // Arid — pale, washed-out, slight haze
    p.horizon  = vec3f(0.85, 0.78, 0.60);
    p.zenith   = vec3f(0.45, 0.60, 0.85);
    p.sunset   = vec3f(1.00, 0.52, 0.10);
    p.night    = vec3f(0.01, 0.01, 0.04);
    p.cloudCov = 0.08;
  } else if (ci < 2.5) {
    // Tropical — deep blue, vivid
    p.horizon  = vec3f(0.38, 0.70, 0.90);
    p.zenith   = vec3f(0.05, 0.18, 0.75);
    p.sunset   = vec3f(1.00, 0.28, 0.50);
    p.night    = vec3f(0.01, 0.01, 0.05);
    p.cloudCov = 0.75;
  } else if (ci < 3.5) {
    // Arctic — cold, pale
    p.horizon  = vec3f(0.70, 0.80, 0.92);
    p.zenith   = vec3f(0.22, 0.38, 0.70);
    p.sunset   = vec3f(0.90, 0.55, 0.60);
    p.night    = vec3f(0.01, 0.01, 0.06);
    p.cloudCov = 0.85;
  } else {
    // Stormy — dark grey-green
    p.horizon  = vec3f(0.30, 0.33, 0.36);
    p.zenith   = vec3f(0.14, 0.16, 0.20);
    p.sunset   = vec3f(0.55, 0.28, 0.10);
    p.night    = vec3f(0.01, 0.01, 0.03);
    p.cloudCov = 0.95;
  }
  return p;
}

// ── Cloud density ─────────────────────────────────────────────────────────────
fn getCloudDensity(view_dir: vec3f, coverage: f32) -> f32 {
  let hf   = clamp(view_dir.y, 0.0, 1.0);
  let mask = smoothstep(0.15, 0.45, hf);
  if (mask < 0.01) { return 0.0; }
  let nd  = normalize(view_dir);
  let uv  = vec2f(nd.x*0.5+0.5, nd.z*0.5+0.5);
  let drift = vec2f(globals.timeOfDay * 2.0, globals.timeOfDay * 1.5);
  let cn  = cloudNoise(uv * 8.0 + vec2f(globals.seed*0.01, globals.seed*0.02) + drift);
  let thresh = mix(0.65, 0.30, coverage);
  return smoothstep(thresh, thresh + 0.25, cn) * mask;
}

// ── Sky color ─────────────────────────────────────────────────────────────────
fn compute_sky_color(world_pos: vec3f, sun_dir: vec3f) -> vec3f {
  let view_dir = normalize(world_pos - globals.cameraPos);
  let tod      = globals.timeOfDay;
  let ci       = clim.index;

  let df  = smoothstep(0.25, 0.35, tod) * (1.0 - smoothstep(0.80, 0.90, tod));
  let sf  = (smoothstep(0.2, 0.3, tod)*(1.0-smoothstep(0.3, 0.4, tod)))
          + (smoothstep(0.7, 0.8, tod)*(1.0-smoothstep(0.8, 0.9, tod)));
  let zenith    = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);
  let sun_angle = dot(view_dir, sun_dir);

  let pal = getSkyPalette(ci, globals.seed);

  var sky = mix(pal.horizon, pal.zenith, zenith);
  sky = mix(sky, pal.sunset, sf * (1.0 - zenith * 0.6));
  sky = mix(pal.night, sky, df);

  // Clouds
  let cd  = getCloudDensity(view_dir, pal.cloudCov * df);
  let cl  = 0.55 + sf * 0.45;
  var cloud_color: vec3f;
  if (ci > 3.5) {
    cloud_color = vec3f(0.24, 0.26, 0.29) * cl;
  } else {
    cloud_color = vec3f(cl) * (1.0 + sf * vec3f(0.9, 0.25, 0.5));
  }
  sky = mix(sky, cloud_color, cd * 0.85);

  // Sun disk
  let sun_disk = smoothstep(0.998, 0.999, sun_angle) * df * 2.0;

  // Moon
  let moon_t    = fract(tod + 0.5);
  let moon_ah   = (moon_t - 0.5) * 3.14159;
  let moon_dir  = normalize(vec3f(-sin(moon_ah), sin(moon_ah*2.0)*0.8, cos(moon_ah)));
  let night_f   = 1.0 - df;
  let moon_dot  = dot(view_dir, moon_dir);
  let moon_disk = smoothstep(0.997, 0.998, moon_dot) * night_f * 0.3;
  let moon_glow = smoothstep(0.990, 0.997, moon_dot) * night_f * 0.08;

  var color = sky + vec3f(sun_disk) + vec3f(moon_disk + moon_glow);

  // Horizon fog — reads globals.fogDensity (set by slider in main.ts)
  let horiz     = 1.0 - abs(view_dir.y);
  let horiz_fog = horiz * horiz * horiz;
  let fog_amt   = clamp(horiz_fog * globals.fogDensity * 8.0, 0.0, 0.85);
  let fog_col   = mix(pal.night,
                      mix(vec3f(0.70, 0.76, 0.85), vec3f(0.95, 0.60, 0.35), sf),
                      df);
  color = mix(color, fog_col, fog_amt);

  return color;
}

// ── Vertex ────────────────────────────────────────────────────────────────────
@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
  var out: VertexOutput;
  out.position  = vec4f(position.xy, position.z, 1.0);
  let world_pos = globals.invViewProj * vec4f(position.xy, position.z, 1.0);
  out.world_pos = world_pos.xyz / world_pos.w;
  return out;
}

// ── Fragment ──────────────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let sky    = compute_sky_color(in.world_pos, globals.sunDir);
  let mapped = sky / (sky + vec3f(1.0));
  let gamma  = pow(mapped, vec3f(1.0 / 2.2));
  return vec4f(gamma, 1.0);
}
