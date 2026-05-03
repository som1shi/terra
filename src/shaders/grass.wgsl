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

@group(0) @binding(0) var<uniform>       globals   : Globals;
@group(0) @binding(1) var<storage, read> instances : array<vec4f>;

struct VertIn {
    @location(0)             localPos : vec3f,
    @location(1)             uv       : vec2f,
    @builtin(instance_index) instIdx  : u32,
};

struct VertOut {
    @builtin(position) clip     : vec4f,
    @location(0)       uv       : vec2f,
    @location(1)       localY   : f32,
    @location(2)       worldPos : vec3f,
};

@vertex
fn vs_main(in: VertIn) -> VertOut {
    let inst = instances[in.instIdx];
    let wp   = inst.xyz;
    let seed = inst.w;

    let r1    = fract(seed * 127.1);
    let r2    = fract(seed * 311.7);
    let scale = mix(4.0, 9.0, r1);
    let yaw   = r2 * 6.28318;

    let cy   = cos(yaw);
    let sy   = sin(yaw);
    let rotY = mat3x3f(
        vec3f( cy, 0.0, sy),
        vec3f(0.0, 1.0, 0.0),
        vec3f(-sy, 0.0, cy),
    );

    let t      = in.localPos.y * in.localPos.y;
    let tapered = vec3f(
        in.localPos.x * (1.0 - t * 0.82),
        in.localPos.y,
        in.localPos.z * (1.0 - t * 0.82),
    );

    let lp = rotY * (tapered * scale);

    let sway    = sin(globals.time * 1.6 + wp.x * 0.04 + wp.z * 0.06) * 0.22 * in.localPos.y;
    let swayVec = vec3f(sway, 0.0, sway * 0.35);

    let worldPos = wp + lp + swayVec;

    var out: VertOut;
    out.clip     = globals.viewProj * vec4f(worldPos, 1.0);
    out.uv       = in.uv;
    out.localY   = in.localPos.y;
    out.worldPos = worldPos;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    let bottomColor = vec3f(0.04, 0.16, 0.03);
    let topColor    = vec3f(0.28, 0.55, 0.12);
    let baseColor   = mix(bottomColor, topColor, in.localY);

    let sunDir    = globals.sunDir;
    let dayFactor = smoothstep(-0.05, 0.15, sunDir.y);

    let sunColor = mix(vec3f(1.5, 0.6, 0.2), vec3f(1.4, 1.2, 0.9),
                       smoothstep(0.0, 0.2, sunDir.y)) * dayFactor;

    let ambient = vec3f(0.09, 0.15, 0.27) * dayFactor
                + vec3f(0.006, 0.006, 0.018) * (1.0 - dayFactor);

    let NdotL   = dot(vec3f(0.0, 1.0, 0.0), sunDir);
    let wrapped = clamp((NdotL + 0.5) / 1.5, 0.0, 1.0);
    let diffuse = wrapped * sunColor * 1.1;

    let color = baseColor * (ambient + diffuse);
    return vec4f(color, 1.0);
}
