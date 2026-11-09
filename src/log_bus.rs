//! In-memory log bus with per-source history and broadcast to subscribers.
//!
//! Sources we use:
//!   - "loca"  — global / proxy events
//!   - <alias> — one stream per loaded backend model

use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::broadcast;

const CAPACITY_PER_SOURCE: usize = 4000;
const BROADCAST_BUF: usize = 4096;

#[derive(Clone)]
pub struct LogEvent {
    pub source: String,
    pub line: String,
}

pub struct LogBus {
    history: Mutex<HashMap<String, VecDeque<String>>>,
    tx: broadcast::Sender<LogEvent>,
}

impl LogBus {
    pub fn new() -> Arc<Self> {
        let (tx, _) = broadcast::channel(BROADCAST_BUF);
        Arc::new(Self {
            history: Mutex::new(HashMap::new()),
            tx,
        })
    }

    pub fn emit(&self, source: &str, msg: &str) {
        let ts = chrono_lite::hms_now();
        let line = format!("{ts}  {}", sanitize(msg));
        {
            let mut hist = self.history.lock();
            let buf = hist
                .entry(source.to_string())
                .or_insert_with(|| VecDeque::with_capacity(CAPACITY_PER_SOURCE));
            if buf.len() == CAPACITY_PER_SOURCE {
                buf.pop_front();
            }
            buf.push_back(line.clone());
        }
        let _ = self.tx.send(LogEvent {
            source: source.to_string(),
            line,
        });
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LogEvent> {
        self.tx.subscribe()
    }

    pub fn history(&self, source: &str) -> Vec<String> {
        self.history
            .lock()
            .get(source)
            .map(|d| d.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn clear_source(&self, source: &str) {
        if let Some(buf) = self.history.lock().get_mut(source) {
            buf.clear();
        }
    }
}

/// Progress bars (e.g. the HF "Fetching N files" bar on mlx's stderr) redraw
/// with `\r`; drawn raw, they smear over the rest of the TUI. Keep the last
/// frame and drop other control characters.
fn sanitize(msg: &str) -> String {
    let last = msg.trim_end_matches('\r').rsplit('\r').next().unwrap_or(msg);
    last.chars().filter(|c| !c.is_control() || *c == '\t').collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_keeps_last_progress_frame() {
        assert_eq!(sanitize("Fetching 9 files:  10%|\rFetching 9 files: 100%|\r"), "Fetching 9 files: 100%|");
        assert_eq!(sanitize("a\x1b[Kb"), "a[Kb");
        assert_eq!(sanitize("plain"), "plain");
    }

    #[test]
    fn emit_writes_to_history() {
        let bus = LogBus::new();
        bus.emit("loca", "hello");
        bus.emit("loca", "world");
        bus.emit("model-a", "x");
        let loca = bus.history("loca");
        assert_eq!(loca.len(), 2);
        assert!(loca[0].contains("hello"));
        assert!(loca[1].contains("world"));
        assert_eq!(bus.history("model-a").len(), 1);
        assert_eq!(bus.history("missing").len(), 0);
    }

    #[test]
    fn clear_source_drops_history_for_that_source_only() {
        let bus = LogBus::new();
        bus.emit("loca", "x");
        bus.emit("a", "y");
        bus.clear_source("loca");
        assert!(bus.history("loca").is_empty());
        assert_eq!(bus.history("a").len(), 1);
    }

    #[test]
    fn history_caps_at_capacity_per_source() {
        let bus = LogBus::new();
        // emit 1 more than capacity; oldest should drop
        for i in 0..(CAPACITY_PER_SOURCE + 1) {
            bus.emit("loca", &format!("line {i}"));
        }
        let hist = bus.history("loca");
        assert_eq!(hist.len(), CAPACITY_PER_SOURCE);
        // first line should be "line 1", not "line 0"
        assert!(hist[0].contains("line 1"));
        assert!(hist[hist.len() - 1].contains(&format!("line {}", CAPACITY_PER_SOURCE)));
    }

    #[tokio::test]
    async fn subscribers_receive_emitted_events() {
        let bus = LogBus::new();
        let mut rx = bus.subscribe();
        bus.emit("loca", "a");
        bus.emit("model-x", "b");
        let e1 = rx.recv().await.unwrap();
        let e2 = rx.recv().await.unwrap();
        assert_eq!(e1.source, "loca");
        assert!(e1.line.contains("a"));
        assert_eq!(e2.source, "model-x");
        assert!(e2.line.contains("b"));
    }

    #[test]
    fn hms_now_format() {
        let s = chrono_lite::hms_now();
        // HH:MM:SS
        assert_eq!(s.len(), 8);
        assert_eq!(s.as_bytes()[2], b':');
        assert_eq!(s.as_bytes()[5], b':');
    }
}

mod chrono_lite {
    //! Tiny "what time is it now (HH:MM:SS)?" helper without pulling chrono in.
    use std::time::{SystemTime, UNIX_EPOCH};

    pub fn hms_now() -> String {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        // Local time offset is not easily computed without tz data; just use UTC.
        // For our log lines this is fine — they're for short-term tailing.
        let day_secs = secs % 86_400;
        let h = day_secs / 3600;
        let m = (day_secs % 3600) / 60;
        let s = day_secs % 60;
        format!("{h:02}:{m:02}:{s:02}")
    }
}
