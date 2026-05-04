use std::collections::HashMap;
use std::fs;
use std::path::Path;

use super::models::*;

pub fn load_atlas_from_directory<P: AsRef<Path>>(atlas_path: P) -> Result<AtlasMetadata, Box<dyn std::error::Error>> {
    let root_path = atlas_path.as_ref();
    let mut metadata = AtlasMetadata {
        samples: HashMap::new(),
        models: HashMap::new(),
    };

    let models_dir = root_path.join("models");
    if models_dir.exists() {
        for task_entry in fs::read_dir(models_dir)? {
            let task_dir = task_entry?.path();
            if !task_dir.is_dir() {
                continue;
            }
            for arch_entry in fs::read_dir(task_dir)? {
                let arch_dir = arch_entry?.path();
                if !arch_dir.is_dir() {
                    continue;
                }
                for json_entry in fs::read_dir(arch_dir)? {
                    let json_path = json_entry?.path();
                    if json_path.extension().and_then(|s| s.to_str()) == Some("json") {
                        if let Ok(model) = load_json::<Model>(&json_path) {
                            metadata.models.insert(model.id.clone(), model);
                        }
                    }
                }
            }
        }
    }

    let samples_dir = root_path.join("samples");
    if !samples_dir.exists() {
        return Ok(metadata);
    }

    for sample_entry in fs::read_dir(samples_dir)? {
        let sample_dir = sample_entry?.path();
        if !sample_dir.is_dir() {
            continue;
        }

        let sample_id = sample_dir.file_name().unwrap().to_string_lossy().to_string();
        let sample_file = sample_dir.join(format!("{}.json", sample_id));

        if !sample_file.exists() {
            continue;
        }

        let sample = load_json::<Sample>(&sample_file)?;
        let scans = load_json_files_from_dir::<Scan>(&sample_dir.join("scans"));
        let volumes = load_json_files_from_dir::<Volume>(&sample_dir.join("volumes"));
        let segments = load_json_files_from_dir::<Segment>(&sample_dir.join("segments"));

        let atlas_sample = AtlasSample {
            sample,
            scans,
            volumes,
            segments,
        };

        metadata.samples.insert(sample_id, atlas_sample);
    }

    Ok(metadata)
}

fn load_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let item = serde_json::from_str(&content)?;
    Ok(item)
}

fn load_json_files_from_dir<T: serde::de::DeserializeOwned + Clone>(dir: &Path) -> HashMap<String, T>
where
    T: HasId,
{
    let mut items = HashMap::new();

    if !dir.exists() {
        return items;
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                if let Ok(item) = load_json::<T>(&path) {
                    items.insert(item.id().to_string(), item);
                }
            }
        }
    }

    items
}

trait HasId {
    fn id(&self) -> &str;
}

impl HasId for Scan {
    fn id(&self) -> &str {
        &self.id
    }
}

impl HasId for Volume {
    fn id(&self) -> &str {
        &self.id
    }
}

impl HasId for Segment {
    fn id(&self) -> &str {
        &self.id
    }
}
