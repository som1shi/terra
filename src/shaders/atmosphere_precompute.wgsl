const R_EARTH    : f32   = 6371.0;
const R_TOP      : f32   = 6471.0;
const H_ATM      : f32   = 100.0;
const H_R        : f32   = 8.0;
const H_M        : f32   = 1.2;
const BETA_R     : vec3f = vec3f(5.802e-3, 13.558e-3, 33.1e-3);
const BETA_M_EXT : f32   = 0.9e-3;
const STEPS      : u32   = 40u;

@group(0) @binding(0) var lut : texture_storage_2d<rgba32float, write>;

fn distToAtmTop(r: f32, mu: f32) -> f32 {
    let disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    return max(0.0, -r * mu + sqrt(max(disc, 0.0)));
}

@compute @workgroup_size(8, 8)
fn compute_transmittance(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(lut);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let u_r  = (f32(id.x) + 0.5) / f32(size.x);
    let u_mu = (f32(id.y) + 0.5) / f32(size.y);

    let r  = R_EARTH + u_r * H_ATM;
    let mu = 2.0 * u_mu - 1.0;

    let disc_ground = r * r * (mu * mu - 1.0) + R_EARTH * R_EARTH;
    if (disc_ground >= 0.0 && mu < 0.0) {
        textureStore(lut, id.xy, vec4f(0.0));
        return;
    }

    let d  = distToAtmTop(r, mu);
    let ds = d / f32(STEPS);

    var sum_r : f32 = 0.0;
    var sum_m : f32 = 0.0;

    for (var i = 0u; i < STEPS; i++) {
        let t   = (f32(i) + 0.5) * ds;
        let r_i = sqrt(r * r + t * t + 2.0 * r * mu * t);
        let h_i = max(r_i - R_EARTH, 0.0);
        sum_r += exp(-h_i / H_R) * ds;
        sum_m += exp(-h_i / H_M) * ds;
    }

    let T_r = exp(-BETA_R * sum_r);
    let T_m = exp(-BETA_M_EXT * sum_m);

    textureStore(lut, id.xy, vec4f(T_r, T_m));
}
