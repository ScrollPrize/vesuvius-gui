#![allow(dead_code)]
use super::{default_cache_dir_for_url, parse_json, ZarrArray, ZarrContext};
use ehttp::Request;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct OmeMultiScale {
    pub axes: Vec<OmeAxis>,
    pub datasets: Vec<OmeDataset>,
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeAxis {
    pub name: String,
    pub r#type: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmeDataset {
    pub coordinate_transformations: Vec<OmeCoordinateTransformation>,
    pub path: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OmeCoordinateTransformation {
    #[allow(non_camel_case_types)]
    scale(OmeScale),
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeScale {
    pub scale: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OmeZarrAttrs {
    pub multiscales: Vec<OmeMultiScale>,
}

impl OmeZarrAttrs {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let multiscales: Vec<OmeMultiScale> = serde_json::from_str(json)?;
        Ok(OmeZarrAttrs { multiscales })
    }
}

#[derive(Debug, Clone)]
pub struct OmeZarr {
    attrs: OmeZarrAttrs,
}

pub struct OmeZarrContext {
    pub ome_zarr: OmeZarr,
    pub cache_missing: bool,
    pub zarr_contexts: Vec<ZarrContext<3>>, // TODO: make generic
}

impl OmeZarrContext {
    pub fn from_url(url: &str, local_cache_dir: &str) -> Self {
        let attrs = Self::load_attrs(url, local_cache_dir);

        let ome_zarr = OmeZarr { attrs };
        let zarr_contexts = ome_zarr.attrs.multiscales[0]
            .datasets
            .iter()
            .map(|dataset| {
                let url_path = format!("{}/{}", url, dataset.path);
                let cache_path = format!("{}/{}", local_cache_dir, dataset.path);
                ZarrArray::from_url(&url_path, &cache_path).into_ctx().into_ctx()
            })
            .collect();

        Self {
            ome_zarr,
            zarr_contexts,
            cache_missing: false,
        }
    }
    pub fn from_url_to_default_cache_dir(url: &str) -> Self {
        let url = if url.ends_with("/") { &url[..url.len() - 1] } else { url };
        Self::from_url(url, &default_cache_dir_for_url(url))
    }
    pub fn from_path(path: &str) -> Self {
        let attrs = {
            let target_file = format!("{}/.zattrs", path);
            let zarray = std::fs::read_to_string(&target_file).unwrap();
            parse_json::<OmeZarrAttrs>(&zarray, &target_file)
        };

        let ome_zarr = OmeZarr { attrs };
        let zarr_contexts = ome_zarr.attrs.multiscales[0]
            .datasets
            .iter()
            .map(|dataset| {
                let path = format!("{}/{}", path, dataset.path);
                ZarrArray::from_path(&path).into_ctx().into_ctx()
            })
            .collect();

        Self {
            ome_zarr,
            zarr_contexts,
            cache_missing: false,
        }
    }

    fn load_attrs(url: &str, local_cache_dir: &str) -> OmeZarrAttrs {
        let target_file = format!("{}/.zattrs", local_cache_dir);
        if !std::path::Path::new(&target_file).exists() {
            let data = ehttp::fetch_blocking(&Request::get(&format!("{}/.zattrs", url)))
                .unwrap()
                .bytes
                .to_vec();
            std::fs::create_dir_all(std::path::Path::new(&target_file).parent().unwrap()).unwrap();
            std::fs::write(&target_file, &data).unwrap();
        }

        let zarray = std::fs::read_to_string(&target_file).unwrap();
        parse_json::<OmeZarrAttrs>(&zarray, &target_file)
    }

    pub fn get(&self, xyz: [usize; 3], scale: u8) -> u8 {
        let max = self.zarr_contexts.len() as u8 - 1;
        for s in scale.min(max)..=max {
            let scaled_xyz = xyz.iter().map(|&x| x >> s).collect::<Vec<usize>>();
            let v = self.zarr_contexts[s as usize].get(scaled_xyz.try_into().unwrap());
            if let Some(v) = v {
                return v;
            }
        }
        0
    }
    pub fn get_interpolated(&self, xyz: [f64; 3], scale: u8) -> u8 {
        let max = self.zarr_contexts.len() as u8 - 1;
        for s in scale.min(max)..=max {
            let scaled_xyz = xyz.iter().map(|&x| x / (1 << s) as f64).collect::<Vec<f64>>();
            let v = self.zarr_contexts[s as usize].get_interpolated(scaled_xyz.try_into().unwrap());
            if let Some(v) = v {
                return v;
            }
        }
        0
    }
}
