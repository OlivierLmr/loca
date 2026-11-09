//! Library crate. Modules are exposed for integration tests; the binary
//! at `src/main.rs` consumes them via `loca::*`.

pub mod assemble;
pub mod backend;
pub mod config;
pub mod init;
pub mod log_bus;
pub mod manager;
pub mod paths;
pub mod proxy;
pub mod stats;
pub mod sysmem;
pub mod tui;
