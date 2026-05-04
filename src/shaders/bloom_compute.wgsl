@group(0) @binding(0) var srcTex : texture_2d<f32>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn bloom_threshold(@builtin(global_invocation_id) id: vec3u) {
    let dst_size = textureDimensions(dstTex);
    if (id.x >= dst_size.x || id.y >= dst_size.y) { return; }

    let sx = id.x * 2u;
    let sy = id.y * 2u;
    let c  = (textureLoad(srcTex, vec2u(sx,    sy),    0).rgb
            + textureLoad(srcTex, vec2u(sx+1u,  sy),    0).rgb
            + textureLoad(srcTex, vec2u(sx,    sy+1u),  0).rgb
            + textureLoad(srcTex, vec2u(sx+1u,  sy+1u), 0).rgb) * 0.25;

    let lum  = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    let thr  = 0.80;
    let knee = 0.3;
    let bright = max(c - thr, vec3f(0.0)) / max(lum - thr + 0.001, 0.001)
               * max(lum - thr + knee, 0.0) / max(lum + knee, 0.001);
    textureStore(dstTex, id.xy, vec4f(bright, 1.0));
}

@group(1) @binding(0) var blurSrc : texture_2d<f32>;
@group(1) @binding(1) var blurDst : texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var blurSmp : sampler;
@group(1) @binding(3) var<uniform> blurOffset : vec4f;

@compute @workgroup_size(8, 8)
fn blur_downsample(@builtin(global_invocation_id) id: vec3u) {
    let dst_size = textureDimensions(blurDst);
    if (id.x >= dst_size.x || id.y >= dst_size.y) { return; }
    let src_size = vec2f(textureDimensions(blurSrc));
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dst_size);
    let d  = 1.0 / src_size;
    let o  = blurOffset.x + 0.5;
    let c  = textureSampleLevel(blurSrc, blurSmp, uv, 0.0).rgb * 4.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x,  d.y) * o, 0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x,  d.y) * o, 0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x, -d.y) * o, 0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x, -d.y) * o, 0.0).rgb;
    textureStore(blurDst, id.xy, vec4f(c / 8.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn blur_upsample(@builtin(global_invocation_id) id: vec3u) {
    let dst_size = textureDimensions(blurDst);
    if (id.x >= dst_size.x || id.y >= dst_size.y) { return; }
    let src_size = vec2f(textureDimensions(blurSrc));
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dst_size);
    let d  = 1.0 / src_size;
    let o  = blurOffset.x + 0.5;
    let c  = textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x * 2.0, 0.0) * o,  0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x,  d.y) * o,        0.0).rgb * 2.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( 0.0,  d.y * 2.0) * o,  0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x,  d.y) * o,        0.0).rgb * 2.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x * 2.0, 0.0) * o,   0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( d.x, -d.y) * o,        0.0).rgb * 2.0
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f( 0.0, -d.y * 2.0) * o,  0.0).rgb
           + textureSampleLevel(blurSrc, blurSmp, uv + vec2f(-d.x, -d.y) * o,        0.0).rgb * 2.0;
    textureStore(blurDst, id.xy, vec4f(c / 12.0, 1.0));
}
