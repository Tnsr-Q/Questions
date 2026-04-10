use serde::{Deserialize, Serialize};

/// 14D symmetric metric tile (row-major)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricTile {
    pub chart_id: u32,
    pub origin: [f32; 14],
    pub g: [[f32; 14]; 14],
    pub mask: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Embedding {
    pub chart_id: u32,
    /// 3x14 Jacobian J such that h = J^T g J
    pub j: [[f32; 14]; 3],
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measure {
    pub kind: MeasureKind,
    pub values: [f32; 4],
    pub uncertainty: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MeasureKind { Length, Angle, Sectional }
