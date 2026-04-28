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
};

@group(0) @binding(0) var<uniform>       globals : Globals;
@group(0) @binding(1) var<storage, read> birds   : array<vec4f>;

struct VertIn {
    // xyz = local-space position, w = flap weight (0=body, 0.4=elbow, 1=tip)
    @location(0)             vertex  : vec4f,
    @builtin(instance_index) instIdx : u32,
};

struct VertOut {
    @builtin(position) clip     : vec4f,
    @location(0)       worldPos : vec3f,
    @location(1)       normal   : vec3f,
    @location(2)       flapW    : f32,
};

@vertex
fn vs_main(in: VertIn) -> VertOut {
    let i   = in.instIdx * 2u;
    let pos = birds[i].xyz;
    let fwd = normalize(birds[i + 1u].xyz);

    // Build orthonormal frame
    let world_up = vec3f(0.0, 1.0, 0.0);
    let alt_up   = vec3f(1.0, 0.0, 0.0);
    let ref_up   = select(world_up, alt_up, abs(dot(fwd, world_up)) > 0.99);
    let right    = normalize(cross(ref_up, fwd));
    let up       = normalize(cross(fwd, right));

    // Flap: ~9 Hz, per-bird phase, wings sweep slightly forward on up-stroke
    let flapPhase = globals.time * 9.0 + f32(in.instIdx) * 0.713;
    let flapSin   = sin(flapPhase);
    let flapY     =  flapSin              * in.vertex.w * 0.28;
    let flapZ     = -cos(flapPhase) * 0.5 * in.vertex.w * 0.08;

    let lp       = vec3f(in.vertex.x, in.vertex.y + flapY, in.vertex.z + flapZ);
    let scale    = 11.0;
    let worldPos = pos
                 + right * (lp.x * scale)
                 + up    * (lp.y * scale)
                 + fwd   * (lp.z * scale);

    // Per-vertex approximate normal: tilt outward on wings (dihedral ~ 16°).
    // Wings are angled up by 0.28 y per 1.0 x, so normal tilts inward by that ratio.
    let dihTilt      = -in.vertex.x * 0.28;          // inward tilt in local X
    let local_norm   = normalize(vec3f(dihTilt, 1.0, 0.0));
    let world_norm   = normalize(right * local_norm.x + up * local_norm.y);

    var out: VertOut;
    out.clip     = globals.viewProj * vec4f(worldPos, 1.0);
    out.worldPos = worldPos;
    out.normal   = world_norm;
    out.flapW    = in.vertex.w;
    return out;
}

@fragment
fn fs_main(in: VertOut) -> @location(0) vec4f {
    let sunDir    = globals.sunDir;
    let dayFactor = smoothstep(-0.05, 0.2, sunDir.y);

    // Body centre = dark charcoal; wing tips = slightly warmer brown
    let bodyColor = vec3f(0.07, 0.06, 0.08);   // dark with faint purple (starling)
    let wingColor = vec3f(0.16, 0.13, 0.11);   // slightly warmer at tips
    let base      = mix(bodyColor, wingColor, smoothstep(0.0, 1.0, in.flapW));

    // Diffuse – use abs so both faces lit (bird is thin)
    let NdotL   = abs(dot(normalize(in.normal), sunDir));
    let diffuse = NdotL * dayFactor * 0.65;
    let ambient = mix(0.20, 0.35, dayFactor);  // boosted so birds read in any lighting

    // Blinn-Phong glimmer (iridescent starling sheen)
    let viewDir = normalize(globals.cameraPos - in.worldPos);
    let halfVec = normalize(sunDir + viewDir);
    let spec    = pow(max(0.0, abs(dot(normalize(in.normal), halfVec))), 20.0)
                  * 0.30 * dayFactor;
    let specCol = vec3f(0.4, 0.8, 0.5) * spec;  // greenish iridescence

    return vec4f(base * (ambient + diffuse) + specCol, 1.0);
}
