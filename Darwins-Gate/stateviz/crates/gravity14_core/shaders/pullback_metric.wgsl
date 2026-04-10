struct JacobianRow { v: array<f32, 14>; };
struct MetricRow   { v: array<f32, 14>; };

struct EmbeddingJ {
  j0: JacobianRow,
  j1: JacobianRow,
  j2: JacobianRow,
};

struct MetricG {
  g: array<MetricRow, 14>;
};

struct OutH {
  // 3x3 row-major, padded to 16-byte rows in std430-like layout
  h00: f32; h01: f32; h02: f32; _pad0: f32;
  h10: f32; h11: f32; h12: f32; _pad1: f32;
  h20: f32; h21: f32; h22: f32; _pad2: f32;
};

@group(0) @binding(0) var<storage, read>  G: MetricG;
@group(0) @binding(1) var<storage, read>  J: EmbeddingJ;
@group(0) @binding(2) var<storage, read_write> H: array<OutH>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Single-tile illustrative kernel (expand to many tiles in real build)
  var tmp: array<array<f32, 14>, 3>;
  for (var a: u32 = 0u; a < 3u; a = a + 1u) {
    for (var i: u32 = 0u; i < 14u; i = i + 1u) {
      var s: f32 = 0.0;
      for (var k: u32 = 0u; k < 14u; k = k + 1u) {
        let Jak = select(J.j0.v[k], select(J.j2.v[k], J.j1.v[k], a == 1u), a == 2u);
        s = s + Jak * G.g[k].v[i];
      }
      tmp[a][i] = s;
    }
  }
  // h = tmp * J
  var h: array<array<f32, 3>, 3>;
  for (var a: u32 = 0u; a < 3u; a = a + 1u) {
    for (var b: u32 = 0u; b < 3u; b = b + 1u) {
      var s: f32 = 0.0;
      for (var i: u32 = 0u; i < 14u; i = i + 1u) {
        let Jbi = select(J.j0.v[i], select(J.j2.v[i], J.j1.v[i], b == 1u), b == 2u);
        s = s + tmp[a][i] * Jbi;
      }
      h[a][b] = s;
    }
  }
  H[gid.x].h00 = h[0][0]; H[gid.x].h01 = h[0][1]; H[gid.x].h02 = h[0][2];
  H[gid.x].h10 = h[1][0]; H[gid.x].h11 = h[1][1]; H[gid.x].h12 = h[1][2];
  H[gid.x].h20 = h[2][0]; H[gid.x].h21 = h[2][1]; H[gid.x].h22 = h[2][2];
}
