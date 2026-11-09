//! System memory snapshot for the TUI header (macOS `sysctl`).
//!
//! When the machine swaps, every model slows down several-fold, which looks
//! exactly like "the model is slow". Showing swap next to the token rates
//! makes that visible.

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SysMem {
    pub swap_used_gb: f64,
    pub swap_total_gb: f64,
    /// macOS "System-wide memory free percentage" (as in `memory_pressure`).
    pub free_pct: Option<u32>,
}

pub async fn read() -> Option<SysMem> {
    let out = tokio::process::Command::new("sysctl")
        .args(["-n", "vm.swapusage", "kern.memorystatus_level"])
        .output()
        .await
        .ok()?;
    parse(&String::from_utf8_lossy(&out.stdout))
}

/// Parses `sysctl -n vm.swapusage kern.memorystatus_level` output:
/// `total = 29696.00M  used = 28080.62M  free = 1615.38M  (encrypted)\n55\n`
pub fn parse(text: &str) -> Option<SysMem> {
    let mut lines = text.lines();
    let swap = lines.next()?;
    let field = |name: &str| -> Option<f64> {
        let rest = swap.split(&format!("{name} = ")).nth(1)?;
        let num = rest.split_whitespace().next()?;
        let (value, unit) = num.split_at(num.len() - 1);
        let value: f64 = value.parse().ok()?;
        Some(match unit {
            "G" => value,
            "M" => value / 1024.0,
            "K" => value / (1024.0 * 1024.0),
            _ => return None,
        })
    };
    Some(SysMem {
        swap_used_gb: field("used")?,
        swap_total_gb: field("total")?,
        free_pct: lines.next().and_then(|l| l.trim().parse().ok()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_sysctl_output() {
        let m = parse("total = 29696.00M  used = 28080.62M  free = 1615.38M  (encrypted)\n55\n")
            .unwrap();
        assert!((m.swap_total_gb - 29.0).abs() < 0.01);
        assert!((m.swap_used_gb - 27.42).abs() < 0.01);
        assert_eq!(m.free_pct, Some(55));
    }

    #[test]
    fn missing_level_is_none() {
        let m = parse("total = 0.00M  used = 0.00M  free = 0.00M\n").unwrap();
        assert_eq!(m.free_pct, None);
        assert_eq!(m.swap_used_gb, 0.0);
    }
}
