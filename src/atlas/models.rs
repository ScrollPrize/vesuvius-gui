use crate::volume::AffineTransform;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Clone, Deserialize)]
pub struct AccessRoot {
    #[serde(rename = "type")]
    pub root_type: String,
    pub url: String,
    pub usage: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataOrigin {
    pub path: String,
    pub access_roots: Vec<AccessRoot>,
    pub copy_info: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataEntry {
    #[serde(rename = "type")]
    pub data_type: String,
    pub origins: Vec<DataOrigin>,
    pub properties: Option<serde_json::Value>,
    pub parameters: Option<serde_json::Value>,
    pub creation_info: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Creation {
    pub date: String,
    pub process: String,
    pub metadata: Option<serde_json::Value>,
    pub derived_from: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TransformTo {
    pub to_volume_id: String,
    pub matrix: Vec<Vec<f64>>,
    pub derivation_path: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VolumeTransforms {
    pub from_volume_id: String,
    pub transforms: Vec<TransformTo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SampleProperties {
    pub volume_transforms: Option<Vec<VolumeTransforms>>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Sample {
    pub id: String,
    pub canonical_volume_id: Option<String>,
    pub properties: Option<SampleProperties>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScanProperties {
    #[serde(rename = "energy_keV")]
    pub energy_kev: Option<f64>,
    pub detector_distance_mm: Option<f64>,
    pub pixel_size_um: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Scan {
    pub id: String,
    pub sample_id: String,
    pub long_id: Option<String>,
    pub creation: Creation,
    pub properties: Option<ScanProperties>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VolumeProperties {
    #[serde(rename = "energy_keV")]
    pub energy_kev: Option<f64>,
    pub detector_distance_mm: Option<f64>,
    pub pixel_size_um: Option<f64>,
    pub data_format: Option<String>,
    pub shape: Option<Vec<usize>>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Volume {
    pub id: String,
    pub sample_id: String,
    pub scan_id: String,
    pub long_id: Option<String>,
    pub suffix: Option<String>,
    pub creation: Creation,
    pub properties: Option<VolumeProperties>,
    #[serde(default)]
    pub data: Vec<DataEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SegmentProperties {
    pub width: usize,
    pub height: usize,
    pub volume_coverage: Option<HashMap<String, serde_json::Value>>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Segment {
    pub id: String,
    pub sample_id: String,
    pub long_id: Option<String>,
    pub suffix: Option<String>,
    pub original_volume_id: String,
    pub creation: Creation,
    pub properties: SegmentProperties,
    #[serde(default)]
    pub data: Vec<DataEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelProperties {
    pub architecture: String,
    pub task: String,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    pub id: String,
    pub long_id: Option<String>,
    pub suffix: Option<String>,
    pub creation: Creation,
    pub properties: ModelProperties,
    #[serde(default)]
    pub data: Vec<DataEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AtlasSample {
    pub sample: Sample,
    #[serde(default)]
    pub scans: HashMap<String, Scan>,
    #[serde(default)]
    pub volumes: HashMap<String, Volume>,
    #[serde(default)]
    pub segments: HashMap<String, Segment>,
}

impl AtlasSample {
    pub fn get_scan(&self, scan_id: &str) -> Option<&Scan> {
        self.scans.get(scan_id)
    }

    pub fn get_volume(&self, volume_id: &str) -> Option<&Volume> {
        self.volumes.get(volume_id)
    }

    pub fn get_segment(&self, segment_id: &str) -> Option<&Segment> {
        self.segments.get(segment_id)
    }

    pub fn find_segments_for_volume(&self, volume_id: &str) -> Vec<&Segment> {
        self.segments
            .values()
            .filter(|s| s.original_volume_id == volume_id)
            .collect()
    }

    pub fn get_transform(&self, source_volume_id: &str, target_volume_id: &str) -> Option<AffineTransform> {
        if source_volume_id == target_volume_id {
            return None;
        }

        if let Some(ref props) = self.sample.properties {
            if let Some(ref transforms) = props.volume_transforms {
                for vt in transforms {
                    if &vt.from_volume_id == source_volume_id {
                        for transform_to in &vt.transforms {
                            if &transform_to.to_volume_id == target_volume_id {
                                return Some(AffineTransform::from_vec(&transform_to.matrix));
                            }
                        }
                    }
                }
            }
        }

        None
    }

    pub fn get_volumes_for_segment(&self, segment_id: &str) -> Vec<(String, &Volume, bool)> {
        let segment = match self.get_segment(segment_id) {
            Some(s) => s,
            None => return Vec::new(),
        };

        let mut result = Vec::new();

        for (vol_id, volume) in &self.volumes {
            let has_coverage = segment
                .properties
                .volume_coverage
                .as_ref()
                .map(|coverage| coverage.contains_key(vol_id))
                .unwrap_or(false);

            let can_render = vol_id == &segment.original_volume_id
                || has_coverage
                || self.get_transform(&segment.original_volume_id, vol_id).is_some();

            if can_render {
                result.push((vol_id.clone(), volume, has_coverage));
            }
        }

        result
    }

    pub fn has_coverage(&self, segment_id: &str, volume_id: &str) -> bool {
        if let Some(segment) = self.get_segment(segment_id) {
            if &segment.original_volume_id == volume_id {
                return true;
            }
            if let Some(ref coverage) = segment.properties.volume_coverage {
                return coverage.contains_key(volume_id);
            }
        }
        false
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AtlasMetadata {
    #[serde(default)]
    pub samples: HashMap<String, AtlasSample>,
    #[serde(default)]
    pub models: HashMap<String, Model>,
}

fn get_standard_url(data: &[DataEntry], data_type: &str) -> Option<String> {
    for data_entry in data {
        if data_entry.data_type == data_type {
            for origin in &data_entry.origins {
                for access_root in &origin.access_roots {
                    if access_root.usage == "standard" {
                        return Some(format!(
                            "https://data.aws.ash2txt.org/samples/{}",
                            origin.path.trim_start_matches('/')
                        ));
                    }
                }
            }
        }
    }
    None
}

impl Segment {
    pub fn get_obj_url(&self) -> Option<String> {
        get_standard_url(&self.data, "obj")
    }
}

impl Volume {
    pub fn get_ome_zarr_url(&self) -> Option<String> {
        get_standard_url(&self.data, "ome-zarr")
    }
}

impl AtlasMetadata {
    pub fn get_sample(&self, sample_id: &str) -> Option<&AtlasSample> {
        self.samples.get(sample_id)
    }

    pub fn sample_ids(&self) -> Vec<&str> {
        self.samples.keys().map(|s| s.as_str()).collect()
    }

    pub fn get_data_urls(&self, data_entry: &DataEntry, usage: &str) -> Vec<String> {
        data_entry
            .origins
            .iter()
            .flat_map(|origin| {
                origin
                    .access_roots
                    .iter()
                    .filter(|root| root.usage == usage)
                    .map(move |root| {
                        format!(
                            "{}/{}",
                            root.url.trim_end_matches('/'),
                            origin.path.trim_start_matches('/')
                        )
                    })
            })
            .collect()
    }
}
