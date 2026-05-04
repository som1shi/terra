@group(0) @binding(0) var hdrTex : texture_2d<f32>;
@group(0) @binding(1) var bloomTex : texture_2d<f32>;
@group(0) @binding(2) var smp : sampler;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
    var p = array<vec2f, 3>(vec2f(-1, -3), vec2f(-1, 1), vec2f(3, 1));
    var out: VOut;
    out.pos = vec4f(p[vi], 0.0, 1.0);
    out.uv  = p[vi] * vec2f(0.5, -0.5) + 0.5;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4f {
    let hdr   = textureSampleLevel(hdrTex, smp, in.uv, 0.0).rgb;
    let bloom = textureSampleLevel(bloomTex, smp, in.uv, 0.0).rgb;
    let color = clamp(hdr + bloom * 0.18, vec3f(0.0), vec3f(1.0));
    return vec4f(color, 1.0);
}
