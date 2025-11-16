use std::{
    fmt, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use serde::{Serialize, de::DeserializeOwned};

/// File system errors surfaced to GUI tooling.
#[derive(Debug)]
pub enum ProjectIoError {
    Missing {
        path: PathBuf,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Corrupt {
        path: PathBuf,
        source: serde_json::Error,
    },
    Serialize {
        path: PathBuf,
        source: serde_json::Error,
    },
}

impl fmt::Display for ProjectIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProjectIoError::Missing { path } => {
                write!(f, "missing file: {}", path.display())
            }
            ProjectIoError::Io { path, source } => {
                write!(f, "I/O error for {}: {}", path.display(), source)
            }
            ProjectIoError::Corrupt { path, source } => {
                write!(f, "failed to parse {}: {}", path.display(), source)
            }
            ProjectIoError::Serialize { path, source } => {
                write!(f, "failed to serialize {}: {}", path.display(), source)
            }
        }
    }
}

impl std::error::Error for ProjectIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProjectIoError::Missing { .. } => None,
            ProjectIoError::Io { source, .. } => Some(source),
            ProjectIoError::Corrupt { source, .. } => Some(source),
            ProjectIoError::Serialize { source, .. } => Some(source),
        }
    }
}

/// Convenience future that reads JSON and returns the parsed payload.
pub async fn read_json_file<T>(path: impl AsRef<Path>) -> Result<T, ProjectIoError>
where
    T: DeserializeOwned,
{
    read_json_inner(path.as_ref())
}

/// Blocking helper mirroring [`read_json_file`].
pub fn read_json_file_blocking<T>(path: impl AsRef<Path>) -> Result<T, ProjectIoError>
where
    T: DeserializeOwned,
{
    read_json_inner(path.as_ref())
}

fn read_json_inner<T>(path: &Path) -> Result<T, ProjectIoError>
where
    T: DeserializeOwned,
{
    let data = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            return Err(ProjectIoError::Missing {
                path: path.to_path_buf(),
            });
        }
        Err(err) => {
            return Err(ProjectIoError::Io {
                path: path.to_path_buf(),
                source: err,
            });
        }
    };

    serde_json::from_str(&data).map_err(|source| ProjectIoError::Corrupt {
        path: path.to_path_buf(),
        source,
    })
}

/// Asynchronously write a JSON file, creating parent directories when needed.
pub async fn write_json_file<T>(path: impl AsRef<Path>, value: &T) -> Result<(), ProjectIoError>
where
    T: Serialize,
{
    write_json_inner(path.as_ref(), value)
}

/// Blocking helper mirroring [`write_json_file`].
pub fn write_json_file_blocking<T>(path: impl AsRef<Path>, value: &T) -> Result<(), ProjectIoError>
where
    T: Serialize,
{
    write_json_inner(path.as_ref(), value)
}

fn write_json_inner<T>(path: &Path, value: &T) -> Result<(), ProjectIoError>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        if let Err(err) = fs::create_dir_all(parent) {
            return Err(ProjectIoError::Io {
                path: parent.to_path_buf(),
                source: err,
            });
        }
    }

    let payload =
        serde_json::to_string_pretty(value).map_err(|source| ProjectIoError::Serialize {
            path: path.to_path_buf(),
            source,
        })?;

    fs::write(path, payload).map_err(|source| ProjectIoError::Io {
        path: path.to_path_buf(),
        source,
    })
}
