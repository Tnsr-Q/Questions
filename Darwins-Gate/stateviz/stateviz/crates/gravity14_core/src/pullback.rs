use crate::types::{MetricTile, Embedding};

/// Compute pullback metric h = J^T g J (3x3), returned as row-major [9].
pub fn pullback_metric(tile: &MetricTile, emb: &Embedding) -> [f32; 9] {
    let mut tmp = [[0.0f32; 14]; 3];
    // tmp = J^T g (J is 3x14, g is 14x14)
    for a in 0..3 {
        for i in 0..14 {
            let mut s = 0.0;
            for k in 0..14 { s += emb.j[a][k] * tile.g[k][i]; }
            tmp[a][i] = s;
        }
    }
    // h = tmp * J  => 3x14 times 14x3
    let mut h = [0.0f32; 9];
    for a in 0..3 {
        for b in 0..3 {
            let mut s = 0.0;
            for i in 0..14 { s += tmp[a][i] * emb.j[b][i]; }
            h[a*3 + b] = s;
        }
    }
    h
}
