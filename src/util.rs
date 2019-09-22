use std::error::Error;

pub fn download(url: &str, destination: &str) -> Result<(), Box<Error>> {
    let path = std::path::Path::new(destination);
    if path.exists() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut resp = reqwest::get(url)?;
    if !resp.status().is_success() {
        bail!("unexpected status code: {}", resp.status());
    }
    let mut file = std::fs::File::create(path)?;
    std::io::copy(&mut resp, &mut file)?;
    Ok(())
}

