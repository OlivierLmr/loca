//! Per-request speed stats, read off the SSE stream mlx_lm.server sends back.
//!
//! mlx_lm.server gives us everything we need without extra calls:
//!   - `: keepalive N/T` comments while it processes the prompt, where T is
//!     the number of prompt tokens *not* served from its prompt cache;
//!   - one `data:` chunk per generated token, with the text in
//!     `delta.reasoning` while the model thinks and `delta.content` (or
//!     `delta.tool_calls`) once it answers;
//!   - a final usage chunk (`prompt_tokens`, `completion_tokens`,
//!     `cached_tokens`) when the request has `stream_options.include_usage`.
//!     The proxy turns that on for every streaming request and drops the
//!     chunk again if the client didn't ask for it.

use bytes::Bytes;
use serde_json::Value;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Tokens arriving within this window count toward the live generation rate.
const RECENT_WINDOW: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Sent upstream, nothing back yet (waiting behind another request, or
    /// mlx hasn't started on the prompt).
    Waiting,
    /// Processing the prompt (prefill).
    Prompt,
    Thinking,
    Answering,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::Waiting => "waiting",
            Phase::Prompt => "prompt",
            Phase::Thinking => "thinking",
            Phase::Answering => "answering",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub cached_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct RequestStats {
    pub started: Instant,
    pub phase: Phase,
    /// First `keepalive` comment: mlx started on the prompt.
    pub prompt_started: Option<Instant>,
    /// Prompt tokens mlx actually has to process (excludes cached ones).
    pub prompt_to_process: u64,
    pub prompt_processed: u64,
    pub first_token: Option<Instant>,
    pub last_token: Option<Instant>,
    pub thinking_tokens: u64,
    pub answer_tokens: u64,
    recent: VecDeque<Instant>,
    pub usage: Option<Usage>,
    /// `data: [DONE]` seen, i.e. the backend finished normally.
    pub completed: bool,
    pub ended: Option<Instant>,
}

impl RequestStats {
    pub fn new(now: Instant) -> Self {
        Self {
            started: now,
            phase: Phase::Waiting,
            prompt_started: None,
            prompt_to_process: 0,
            prompt_processed: 0,
            first_token: None,
            last_token: None,
            thinking_tokens: 0,
            answer_tokens: 0,
            recent: VecDeque::new(),
            usage: None,
            completed: false,
            ended: None,
        }
    }

    pub fn generated_tokens(&self) -> u64 {
        match self.usage {
            Some(u) => u.completion_tokens,
            None => self.thinking_tokens + self.answer_tokens,
        }
    }

    /// Time spent before mlx started on the prompt.
    pub fn wait(&self) -> Option<Duration> {
        self.prompt_started.or(self.first_token).map(|t| t - self.started)
    }

    /// Time to first token, measured from when the request reached loca.
    pub fn ttft(&self) -> Option<Duration> {
        self.first_token.map(|t| t - self.started)
    }

    /// Prompt processing speed over the uncached prompt tokens.
    pub fn prompt_tps(&self, now: Instant) -> Option<f64> {
        let start = self.prompt_started?;
        let end = self.first_token.unwrap_or(now);
        let processed = if self.first_token.is_some() {
            self.prompt_to_process
        } else {
            self.prompt_processed
        };
        rate(processed, end - start)
    }

    /// Average generation speed since the first token.
    pub fn gen_tps(&self) -> Option<f64> {
        let (first, last) = (self.first_token?, self.last_token?);
        rate(self.generated_tokens().saturating_sub(1), last - first)
    }

    /// Generation speed over the last few seconds.
    pub fn live_tps(&self, now: Instant) -> Option<f64> {
        let oldest = *self.recent.front()?;
        if self.recent.len() < 2 || now - oldest > RECENT_WINDOW * 2 {
            return None;
        }
        rate(self.recent.len() as u64 - 1, *self.recent.back()? - oldest)
    }

    pub fn elapsed(&self, now: Instant) -> Duration {
        self.ended.unwrap_or(now) - self.started
    }

    fn on_keepalive(&mut self, processed: u64, total: u64, now: Instant) {
        if self.prompt_started.is_none() {
            self.prompt_started = Some(now);
        }
        self.prompt_to_process = total;
        self.prompt_processed = processed;
        if self.first_token.is_none() {
            self.phase = Phase::Prompt;
        }
    }

    fn on_token(&mut self, thinking: bool, now: Instant) {
        if self.first_token.is_none() {
            self.first_token = Some(now);
            self.prompt_processed = self.prompt_to_process;
        }
        self.last_token = Some(now);
        if thinking {
            self.thinking_tokens += 1;
            self.phase = Phase::Thinking;
        } else {
            self.answer_tokens += 1;
            self.phase = Phase::Answering;
        }
        self.recent.push_back(now);
        while self.recent.front().is_some_and(|t| now - *t > RECENT_WINDOW) {
            self.recent.pop_front();
        }
    }

    /// Same numbers as the TUI, for `/admin/status`.
    pub fn to_json(&self, now: Instant) -> Value {
        let secs = |d: Option<Duration>| d.map(|d| (d.as_secs_f64() * 10.0).round() / 10.0);
        let round = |r: Option<f64>| r.map(|r| (r * 10.0).round() / 10.0);
        serde_json::json!({
            "phase": if self.ended.is_some() {
                if self.completed { "done" } else { "aborted" }
            } else {
                self.phase.label()
            },
            "elapsed_sec": secs(Some(self.elapsed(now))),
            "wait_sec": secs(self.wait()),
            "ttft_sec": secs(self.ttft()),
            "prompt_tokens": self.usage.map(|u| u.prompt_tokens),
            "cached_prompt_tokens": self.usage.map(|u| u.cached_tokens),
            "uncached_prompt_tokens": self.prompt_to_process,
            "prompt_tps": round(self.prompt_tps(now)),
            "thinking_tokens": self.thinking_tokens,
            "answer_tokens": self.answer_tokens,
            "gen_tps": round(self.gen_tps()),
            "live_gen_tps": round(self.live_tps(now)),
        })
    }

    /// One-line summary for the model log once the request ends.
    pub fn summary(&self, now: Instant) -> String {
        let mut parts = vec![format!(
            "{} in {}",
            if self.completed { "done" } else { "aborted" },
            fmt_dur(self.elapsed(now))
        )];
        if let Some(w) = self.wait().filter(|w| *w >= Duration::from_millis(500)) {
            parts.push(format!("waited {}", fmt_dur(w)));
        }
        let mut prompt = match self.usage {
            Some(u) => format!("prompt {} tok ({} cached)", u.prompt_tokens, u.cached_tokens),
            None => format!("prompt {} tok uncached", self.prompt_to_process),
        };
        if let Some(r) = self.prompt_tps(now) {
            prompt.push_str(&format!(" @ {}", fmt_rate(r)));
        }
        parts.push(prompt);
        if let Some(t) = self.ttft() {
            parts.push(format!("first token {}", fmt_dur(t)));
        }
        parts.push(format!(
            "thinking {} · answer {} tok",
            self.thinking_tokens, self.answer_tokens
        ));
        if let Some(r) = self.gen_tps() {
            parts.push(format!("gen {}", fmt_rate(r)));
        }
        parts.join(" · ")
    }
}

fn rate(tokens: u64, d: Duration) -> Option<f64> {
    let secs = d.as_secs_f64();
    (tokens > 0 && secs > 0.05).then(|| tokens as f64 / secs)
}

pub fn fmt_rate(r: f64) -> String {
    if r >= 100.0 {
        format!("{r:.0} tok/s")
    } else {
        format!("{r:.1} tok/s")
    }
}

pub fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s < 10.0 {
        format!("{s:.1}s")
    } else if s < 60.0 {
        format!("{s:.0}s")
    } else {
        let s = s as u64;
        format!("{}m {}s", s / 60, s % 60)
    }
}

/// Splits the upstream byte stream into SSE events, updates `RequestStats`
/// from each one, and returns the bytes to forward to the client. Only whole
/// events are forwarded, so a usage chunk can be dropped cleanly.
pub struct SseTap {
    buf: Vec<u8>,
    /// Drop the usage chunk: the client didn't ask for it, loca did.
    strip_usage: bool,
}

impl SseTap {
    pub fn new(strip_usage: bool) -> Self {
        Self { buf: Vec::new(), strip_usage }
    }

    pub fn feed(&mut self, chunk: &[u8], stats: &mut RequestStats, now: Instant) -> Bytes {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::with_capacity(self.buf.len());
        let mut start = 0;
        while let Some(pos) = find(&self.buf[start..], b"\n\n") {
            let end = start + pos + 2;
            let event = &self.buf[start..end];
            if process_event(event, stats, now) || !self.strip_usage {
                out.extend_from_slice(event);
            }
            start = end;
        }
        self.buf.drain(..start);
        Bytes::from(out)
    }

    /// Whatever is left once upstream closes (normally nothing).
    pub fn finish(&mut self) -> Bytes {
        Bytes::from(std::mem::take(&mut self.buf))
    }
}

/// Updates stats from one SSE event. Returns false for a usage-only chunk.
fn process_event(event: &[u8], stats: &mut RequestStats, now: Instant) -> bool {
    let Ok(text) = std::str::from_utf8(event) else { return true };
    let mut keep = true;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(": keepalive ") {
            if let Some((p, t)) = rest.trim().split_once('/') {
                if let (Ok(p), Ok(t)) = (p.parse(), t.parse()) {
                    stats.on_keepalive(p, t, now);
                }
            }
        } else if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                stats.completed = true;
                continue;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
            if let Some(u) = v.get("usage").filter(|u| u.is_object()) {
                stats.usage = Some(Usage {
                    prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
                    completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
                    cached_tokens: u["prompt_tokens_details"]["cached_tokens"]
                        .as_u64()
                        .unwrap_or(0),
                });
                if v["choices"].as_array().is_none_or(|c| c.is_empty()) {
                    keep = false;
                }
            }
            let delta = &v["choices"][0]["delta"];
            let non_empty = |k: &str| match &delta[k] {
                Value::String(s) => !s.is_empty(),
                Value::Array(a) => !a.is_empty(),
                _ => false,
            };
            if non_empty("content") || non_empty("tool_calls") {
                stats.on_token(false, now);
            } else if non_empty("reasoning") || non_empty("reasoning_content") {
                stats.on_token(true, now);
            }
        }
    }
    keep
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn tracks_prompt_thinking_answer_and_usage() {
        let t0 = Instant::now();
        let mut s = RequestStats::new(t0);
        let mut tap = SseTap::new(true);

        tap.feed(b": keepalive 0/4000\n\n", &mut s, at(t0, 1000));
        assert_eq!(s.phase, Phase::Prompt);
        tap.feed(b": keepalive 2048/4000\n\n", &mut s, at(t0, 2000));
        assert_eq!(s.prompt_processed, 2048);

        let think = b"data: {\"choices\":[{\"delta\":{\"reasoning\":\"hm\"}}]}\n\n";
        tap.feed(think, &mut s, at(t0, 3000));
        tap.feed(think, &mut s, at(t0, 3100));
        assert_eq!(s.phase, Phase::Thinking);
        let ans = b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
        tap.feed(ans, &mut s, at(t0, 3200));
        assert_eq!(s.phase, Phase::Answering);

        let usage = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5000,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":1000}}}\n\n";
        let fwd = tap.feed(usage, &mut s, at(t0, 3300));
        assert!(fwd.is_empty(), "usage chunk is stripped when the client didn't ask");
        tap.feed(b"data: [DONE]\n\n", &mut s, at(t0, 3300));

        assert!(s.completed);
        assert_eq!((s.thinking_tokens, s.answer_tokens), (2, 1));
        assert_eq!(s.usage.unwrap().cached_tokens, 1000);
        assert_eq!(s.wait(), Some(Duration::from_secs(1)));
        assert_eq!(s.ttft(), Some(Duration::from_secs(3)));
        // 4000 uncached tokens between first keepalive (1s) and first token (3s).
        assert_eq!(s.prompt_tps(at(t0, 9999)), Some(2000.0));
        // 2 tokens after the first, over 0.2s.
        assert!((s.gen_tps().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn forwards_only_whole_events_across_chunk_boundaries() {
        let t0 = Instant::now();
        let mut s = RequestStats::new(t0);
        let mut tap = SseTap::new(false);
        let a = tap.feed(b"data: {\"choices\":[{\"delta\":{\"con", &mut s, t0);
        assert!(a.is_empty());
        let b = tap.feed(b"tent\":\"x\"}}]}\n\ndata: [DO", &mut s, t0);
        assert_eq!(&b[..], b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n");
        assert_eq!(s.answer_tokens, 1);
        assert_eq!(&tap.finish()[..], b"data: [DO");
    }

    #[test]
    fn keeps_usage_chunk_when_client_asked_for_it() {
        let t0 = Instant::now();
        let mut s = RequestStats::new(t0);
        let mut tap = SseTap::new(false);
        let usage = b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n";
        assert_eq!(tap.feed(usage, &mut s, t0).len(), usage.len());
        assert!(s.usage.is_some());
    }

    #[test]
    fn summary_marks_aborted_requests() {
        let t0 = Instant::now();
        let s = RequestStats::new(t0);
        assert!(s.summary(at(t0, 2000)).starts_with("aborted in 2.0s"));
    }
}
