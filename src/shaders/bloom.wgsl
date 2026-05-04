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

@group(0) @binding(0) var<uniform> globals : Globals;
@group(0) @binding(1) var hdrTex : texture_2d<f32>;
@group(0) @binding(2) var bloomTex : texture_2d<f32>;
@group(0) @binding(3) var linearSmp : sampler;

fn aces(x: vec3f) -> vec3f {
    let a = 2.51f; let b = 0.03f; let c = 2.43f; let d = 0.59f; let e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2f, 3>(vec2f(-1, -3), vec2f(-1, 1), vec2f(3, 1));
    var out: VOut;
    out.pos = vec4f(p[vi], 0.0, 1.0);
    out.uv  = p[vi] * vec2f(0.5, -0.5) + 0.5;
    return out;
}

@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bloom_threshold(@builtin(global_invocation_id) id: vec3u) {
    let dst_size = textureDimensions(dstTex);
    if (id.x >= dst_size.x || id.y >= dst_size.y) { return; }

    let sx = id.x * 2u;
    let sy = id.y * 2u;
    let c  = (textureLoad(srcTex, vec2u(sx,   sy),   0).rgb
            + textureLoad(srcTex, vec2u(sx+1u, sy),   0).rgb
            + textureLoad(srcTex, vec2u(sx,   sy+1u), 0).rgb
            + textureLoad(srcTex, vec2u(sx+1u, sy+1u),0).rgb) * 0.25;

    let lum = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    let knee = 0.5;
    let thr  = 0.9;
    let bright = max(c - thr, vec3f(0.0)) / max(lum - thr + 0.001, 0.001) * max(lum - thr + knee, 0.0) / max(lum + knee, 0.001);
    textureStore(dstTex, id.xy, vec4f(bright, 1.0));
}

@group(0) @binding(0) var blurSrc : texture_2d<f32>;
@group(0) @binding(1) var blurDst : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var blurSmp : sampler;
@group(0) @binding(3) var<uniform> blurOffset : f32;

@compute @workgroup_size(8, 8)
fn blur_downsample(@builtin(global_invocation_id) id: vec3u) {
    let dst_size = textureDimensions(blurDst);
    if (id.x >= dst_size.x || id.y >= dst_size.y) { return; }
    let src_size = vec2f(textureDimensions(blurSrc));
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dst_size);
    let d  = 1.0 / src_size;
    let o  = blurOffset + 0.5;
    let c  = textureSampleLevel(blurSrc, blurSmp, uv,                                  0.0).rgb * 4.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x,  d.y) * o,          0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x,  d.y) * o,          0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x, -d.y) * o,          0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x, -d.y) * o,          0.0).rgb;
    textureStore(blurDst, id.xy, vec4f(c / 8.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn blur_upsample(@builtin(global_invocation_id) id: vec3u) {
    let dst_size = textureDimensions(blurDst);
    if (id.x >= dst_size.x || id.y >= dst_size.y) { return; }
    let src_size = vec2f(textureDimensions(blurSrc));
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dst_size);
    let d  = 1.0 / src_size;
    let o  = blurOffset + 0.5;
    let c  = textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x * 2.0, 0.0) * o, 0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x,  d.y) * o,       0.0).rgb * 2.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(0.0,   d.y * 2.0) * o, 0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x,  d.y) * o,       0.0).rgb * 2.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x * 2.0, 0.0) * o,  0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x, -d.y) * o,       0.0).rgb * 2.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(0.0,  -d.y * 2.0) * o, 0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x, -d.y) * o,       0.0).rgb * 2.0;
    textureStore(blurDst, id.xy, vec4f(c / 12.0, 1.0));
}

@fragment
fn fs_composite(in: VOut) -> @location(0) vec4f {
    let hdr   = textureSampleLevel(hdrTex, linearSmp, in.uv, 0.0).rgb;
    let bloom = textureSampleLevel(bloomTex, linearSmp, in.uv, 0.0).rgb;

    var color = hdr + bloom * 0.18;

    color = aces(color);
    color = pow(clamp(color, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));

    let uv  = in.uv;
    let vig = 0.5 + 0.5 * pow(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.15);
    color  *= vig;

    let grey = dot(color, vec3f(0.299, 0.587, 0.114));
    color = mix(color, vec3f(grey), 0.10);
    color = pow(color, vec3f(0.97, 1.0, 1.03));

    return vec4f(color, 1.0);
}
