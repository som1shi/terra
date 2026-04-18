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
};

@group(0) @binding(0) var<uniform> globals : Globals;

// Simple noise function for clouds
fn hash(p: vec2f) -> f32 {
  let q = fract(p * 0.3183099 + vec2f(0.1, 0.1));
  return fract((q.x + q.y) * (q.x + q.y + 17.0));
}

fn noise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);

  let a = hash(i + vec2f(0.0, 0.0));
  let b = hash(i + vec2f(1.0, 0.0));
  let c = hash(i + vec2f(0.0, 1.0));
  let d = hash(i + vec2f(1.0, 1.0));

  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn cloudNoise(p: vec2f) -> f32 {
  return noise(p) * 0.5 + noise(p * 2.0) * 0.25 + noise(p * 4.0) * 0.125;
}

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
  var out: VertexOutput;
  // Use position directly as clip space coordinates
  out.position = vec4f(position.xy, position.z, 1.0);

  // Convert clip space back to world space for sky calculation
  let clip_pos = vec4f(position.xy, position.z, 1.0);
  let world_pos = globals.invViewProj * clip_pos;
  out.world_pos = world_pos.xyz / world_pos.w;

  return out;
}

fn getCloudDensity(view_dir: vec3f, seed: f32) -> f32 {
  // Only show clouds in upper part of sky
  let height_factor = clamp(view_dir.y, 0.0, 1.0);
  let cloud_mask = smoothstep(0.2, 0.5, height_factor);

  // If we're looking too low, no clouds
  if (cloud_mask < 0.01) {
    return 0.0;
  }

  // Use a simpler UV mapping to avoid atan2 discontinuities
  let normalized_dir = normalize(view_dir);
  let sky_uv = vec2f(
    normalized_dir.x * 0.5 + 0.5,
    normalized_dir.z * 0.5 + 0.5
  );

  // Add seed variation to cloud pattern
  let cloud_uv = sky_uv * 8.0 + vec2f(seed * 0.01, seed * 0.02);

  // Move clouds based on time of day, not continuous time
  let time_offset = vec2f(globals.timeOfDay * 2.0, globals.timeOfDay * 1.5);

  // Generate cloud noise
  let clouds = cloudNoise(cloud_uv + time_offset);

  // Make clouds more defined and apply height mask
  return smoothstep(0.45, 0.75, clouds) * cloud_mask;
}

fn compute_sky_color(world_pos: vec3f, sun_dir: vec3f) -> vec3f {
  let view_dir = normalize(world_pos - globals.cameraPos);
  let seed = globals.seed;

  // Use time of day directly for more responsive changes
  let tod = globals.timeOfDay;
  let sun_elevation = sun_dir.y;
  let zenith = clamp(view_dir.y * 0.5 + 0.5, 0.0, 1.0);
  let sun_angle = dot(view_dir, sun_dir);

  // More dramatic time transitions based on time of day
  // Make evening go back to night after sunset
  let morning_rise = smoothstep(0.25, 0.35, tod);
  let evening_fall = 1.0 - smoothstep(0.8, 0.9, tod);
  let day_factor = morning_rise * evening_fall;

  // Sunset/sunrise factor
  let dawn_factor = smoothstep(0.2, 0.3, tod) * (1.0 - smoothstep(0.3, 0.4, tod));
  let dusk_factor = smoothstep(0.7, 0.8, tod) * (1.0 - smoothstep(0.8, 0.9, tod));
  let sunset_factor = dawn_factor + dusk_factor;

  // Simple seed-based color variations
  let hue_shift = sin(seed * 0.01) * 0.05;

  // Sky colors with subtle seed variations
  let night_sky = vec3f(0.01, 0.01, 0.05);
  let day_sky_horizon = vec3f(0.6 + hue_shift, 0.7, 0.9 - hue_shift);
  let day_sky_zenith = vec3f(0.1, 0.3, 0.8 + hue_shift);
  let sunset_sky = vec3f(1.0, 0.3, 0.4);

  // Base day sky
  var sky_color = mix(day_sky_horizon, day_sky_zenith, zenith);

  // Add sunset colors
  sky_color = mix(sky_color, sunset_sky, sunset_factor * (1.0 - zenith * 0.6));

  // Mix with night sky
  sky_color = mix(night_sky, sky_color, day_factor);

  // Add clouds
  let cloud_density = getCloudDensity(view_dir, seed);
  let cloud_lighting = 0.6 + sunset_factor * 0.4; // Clouds get warmer at sunset
  let cloud_color = vec3f(cloud_lighting) * (1.0 + sunset_factor * vec3f(0.8, 0.2, 0.4));

  // Show fewer clouds at night
  let cloud_visibility = day_factor * 0.6 + 0.2;
  sky_color = mix(sky_color, cloud_color, cloud_density * cloud_visibility);

  // Add sun disk
  let sun_disk = smoothstep(0.998, 0.999, sun_angle) * day_factor * 2.0;

  // Calculate moon position - opposite to sun
  // Moon rises when sun sets and vice versa
  let moon_time = tod + 0.5; // Moon is 12 hours offset from sun
  let moon_time_wrapped = fract(moon_time); // Keep in 0-1 range

  // Moon position calculation (opposite arc to sun)
  let moon_angle_h = (moon_time_wrapped - 0.5) * 3.14159; // Horizontal angle
  let moon_elevation = sin(moon_angle_h * 2.0) * 0.8; // Vertical position

  // Moon direction vector (opposite to sun's general direction)
  let moon_dir = vec3f(
    -sin(moon_angle_h),
    moon_elevation,
    cos(moon_angle_h)
  );

  // Calculate moon visibility (only show at night)
  let night_factor = 1.0 - day_factor;
  let moon_angle_dot = dot(view_dir, normalize(moon_dir));

  // Moon disk (smaller and dimmer than sun)
  let moon_disk = smoothstep(0.997, 0.998, moon_angle_dot) * night_factor * 0.3;

  // Moon glow around the disk
  let moon_glow = smoothstep(0.99, 0.997, moon_angle_dot) * night_factor * 0.1;

  return sky_color + vec3f(sun_disk) + vec3f(moon_disk + moon_glow);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
  let sky_color = compute_sky_color(in.world_pos, globals.sunDir);

  // Tone mapping and gamma correction
  let mapped = sky_color / (sky_color + vec3f(1.0));
  let gamma_corrected = pow(mapped, vec3f(1.0 / 2.2));

  return vec4f(gamma_corrected, 1.0);
}