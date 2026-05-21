//! ratatui-based foreground TUI: status bar, models table, model log, loca log.
//!
//! State updates each tick from the BackendManager. Logs stream in via the
//! LogBus broadcast channel. Focus determines which pane handles input and
//! which key hints are shown in the footer.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap},
    Frame, Terminal,
};
use std::collections::HashMap;
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::config::ProxyCfg;
use crate::log_bus::{LogBus, LogEvent};
use crate::manager::BackendManager;
use crate::paths;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pane {
    Models,
    ModelLog,
    LocaLog,
}

impl Pane {
    fn next(self) -> Self {
        match self {
            Pane::Models => Pane::ModelLog,
            Pane::ModelLog => Pane::LocaLog,
            Pane::LocaLog => Pane::Models,
        }
    }
}

struct LogBuffer {
    lines: Vec<String>,
    scroll: usize, // 0 = bottom (auto-scrolled), >0 = N lines above bottom
}

impl LogBuffer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            scroll: 0,
        }
    }
    fn push(&mut self, line: String) {
        self.lines.push(line);
        if self.lines.len() > 8000 {
            // Drop the oldest 1000 to keep memory bounded.
            self.lines.drain(0..1000);
        }
    }
    fn clear(&mut self) {
        self.lines.clear();
        self.scroll = 0;
    }
    fn set_history(&mut self, lines: Vec<String>) {
        self.lines = lines;
        self.scroll = 0;
    }
}

struct AppState {
    mgr: Arc<BackendManager>,
    bus: Arc<LogBus>,
    cfg: ProxyCfg,
    start_time: Instant,
    focus: Pane,
    table_state: TableState,
    aliases: Vec<String>, // ordered list matching the table rows
    /// Tracked alias for the model_log pane (the row under cursor).
    selected_alias: Option<String>,
    /// Tail buffers for each known source.
    logs: HashMap<String, LogBuffer>,
    /// Total request and error counters (snapshot in tick).
    tick_total_req: u64,
    tick_total_err: u64,
    tick_ram_used: u32,
    tick_n_loaded: usize,
    /// Per-row snapshot for the table.
    tick_rows: Vec<RowSnapshot>,
    /// Set true to exit the loop.
    quit: bool,
}

#[derive(Clone)]
struct RowSnapshot {
    alias: String,
    ram_gb: Option<u32>,
    description: String,
    state: &'static str,
    idle: String,
    cached: &'static str,
}

impl AppState {
    fn new(mgr: Arc<BackendManager>, bus: Arc<LogBus>, cfg: ProxyCfg) -> Self {
        let aliases: Vec<String> = mgr.models.keys().cloned().collect();
        let mut table_state = TableState::default();
        if !aliases.is_empty() {
            table_state.select(Some(0));
        }
        let mut logs = HashMap::new();
        logs.insert("loca".to_string(), LogBuffer::new());
        for a in &aliases {
            logs.insert(a.clone(), LogBuffer::new());
        }
        let selected_alias = aliases.first().cloned();
        Self {
            mgr,
            bus,
            cfg,
            start_time: Instant::now(),
            focus: Pane::Models,
            table_state,
            aliases,
            selected_alias,
            logs,
            tick_total_req: 0,
            tick_total_err: 0,
            tick_ram_used: 0,
            tick_n_loaded: 0,
            tick_rows: Vec::new(),
            quit: false,
        }
    }

    fn current_alias(&self) -> Option<String> {
        self.table_state.selected().and_then(|i| self.aliases.get(i).cloned())
    }

    fn move_cursor(&mut self, delta: i32) {
        if self.aliases.is_empty() {
            return;
        }
        let n = self.aliases.len() as i32;
        let i = self.table_state.selected().unwrap_or(0) as i32;
        let new = (i + delta).rem_euclid(n) as usize;
        self.table_state.select(Some(new));
        self.selected_alias = Some(self.aliases[new].clone());
        // Re-render the model_log pane from history for the new alias.
        let hist = self.bus.history(&self.aliases[new]);
        if let Some(buf) = self.logs.get_mut(&self.aliases[new]) {
            buf.set_history(hist);
        }
    }
}

pub async fn run(mgr: Arc<BackendManager>, bus: Arc<LogBus>, cfg: ProxyCfg) -> Result<()> {
    let mut state = AppState::new(mgr.clone(), bus.clone(), cfg);

    // Prime the loca log and the initially selected model log with history.
    if let Some(buf) = state.logs.get_mut("loca") {
        buf.set_history(bus.history("loca"));
    }
    if let Some(alias) = state.selected_alias.clone() {
        let hist = bus.history(&alias);
        if let Some(buf) = state.logs.get_mut(&alias) {
            buf.set_history(hist);
        }
    }

    // Channel for terminal events feeding the main loop.
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<Event>();
    let event_handle = tokio::task::spawn_blocking(move || {
        loop {
            if let Ok(true) = event::poll(Duration::from_millis(100)) {
                if let Ok(ev) = event::read() {
                    if event_tx.send(ev).is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Subscribe to the log bus.
    let mut log_rx = bus.subscribe();

    // Set up the terminal.
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut tick = tokio::time::interval(Duration::from_millis(1000));
    refresh_tick(&mut state).await;
    draw(&mut terminal, &mut state)?;

    while !state.quit {
        tokio::select! {
            biased;
            Some(ev) = event_rx.recv() => {
                if let Event::Key(k) = ev {
                    if k.kind == KeyEventKind::Press {
                        handle_key(&mut state, k).await;
                    }
                }
                draw(&mut terminal, &mut state)?;
            }
            Ok(log) = log_rx.recv() => {
                ingest_log(&mut state, log);
                draw(&mut terminal, &mut state)?;
            }
            _ = tick.tick() => {
                refresh_tick(&mut state).await;
                draw(&mut terminal, &mut state)?;
            }
        }
    }

    // Restore terminal.
    crossterm::execute!(terminal.backend_mut(), crossterm::terminal::LeaveAlternateScreen)?;
    crossterm::terminal::disable_raw_mode()?;
    event_handle.abort();
    Ok(())
}

fn ingest_log(state: &mut AppState, ev: LogEvent) {
    state
        .logs
        .entry(ev.source.clone())
        .or_insert_with(LogBuffer::new)
        .push(ev.line);
}

async fn refresh_tick(state: &mut AppState) {
    let instances = state.mgr.instances().await;
    state.tick_n_loaded = instances.len();
    state.tick_ram_used = state.mgr.ram_used_gb().await;
    state.tick_total_req = 0;
    state.tick_total_err = 0;
    let mut by_alias: HashMap<String, &Arc<crate::backend::BackendInstance>> = HashMap::new();
    for i in &instances {
        by_alias.insert(i.alias.clone(), i);
        state.tick_total_req += i.request_count.load(std::sync::atomic::Ordering::Relaxed);
        state.tick_total_err += i.error_count.load(std::sync::atomic::Ordering::Relaxed);
    }
    let mut rows = Vec::with_capacity(state.aliases.len());
    for alias in &state.aliases {
        let m = state.mgr.models.get(alias).cloned();
        let inst = by_alias.get(alias);
        let cached = if paths::hf_cache_for(m.as_ref().map(|m| m.hf.as_str()).unwrap_or(""))
            .map(|p| p.exists())
            .unwrap_or(false)
        {
            "yes"
        } else {
            "no"
        };
        let (state_str, idle) = if let Some(inst) = inst {
            let s = match inst.state().await {
                crate::backend::BackendState::Stopped => "—",
                crate::backend::BackendState::Loading => "LOADING",
                crate::backend::BackendState::Ready => "READY",
            };
            let idle = match inst.last_request_age() {
                Some(age) => format!("{} / {}", fmt_age(age), fmt_age(inst.idle_timeout())),
                None => format!("— / {}", fmt_age(inst.idle_timeout())),
            };
            (s, idle)
        } else {
            ("—", "—".to_string())
        };
        rows.push(RowSnapshot {
            alias: alias.clone(),
            ram_gb: m.as_ref().and_then(|m| m.ram_gb),
            description: m.as_ref().and_then(|m| m.description.clone()).unwrap_or_default(),
            state: state_str,
            idle,
            cached,
        });
    }
    state.tick_rows = rows;
}

async fn handle_key(state: &mut AppState, k: KeyEvent) {
    // Global key bindings (work in any pane).
    if k.code == KeyCode::Char('q')
        || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL))
    {
        state.mgr.stop_all().await;
        state.quit = true;
        return;
    }
    if k.code == KeyCode::Tab {
        state.focus = state.focus.next();
        return;
    }

    match state.focus {
        Pane::Models => handle_models_key(state, k).await,
        Pane::ModelLog => handle_log_key(state, k, /* model_log */ true),
        Pane::LocaLog => handle_log_key(state, k, /* model_log */ false),
    }
}

async fn handle_models_key(state: &mut AppState, k: KeyEvent) {
    match k.code {
        KeyCode::Up => state.move_cursor(-1),
        KeyCode::Down => state.move_cursor(1),
        KeyCode::Enter => toggle_selected(state).await,
        KeyCode::Char('s') => stop_selected(state).await,
        KeyCode::Char('t') => adjust_timeout(state, 600, "+10m"),
        KeyCode::Char('T') => adjust_timeout(state, 3600, "+1h"),
        KeyCode::Char('r') => reset_timeout(state),
        _ => {}
    }
}

fn handle_log_key(state: &mut AppState, k: KeyEvent, model_log: bool) {
    match k.code {
        KeyCode::Char('c') => {
            let source = if model_log {
                state.selected_alias.clone()
            } else {
                Some("loca".to_string())
            };
            if let Some(s) = source {
                state.bus.clear_source(&s);
                if let Some(buf) = state.logs.get_mut(&s) {
                    buf.clear();
                }
            }
        }
        _ => {}
    }
}

async fn toggle_selected(state: &mut AppState) {
    let Some(alias) = state.current_alias() else { return };
    if let Some(inst) = state.mgr.get(&alias).await {
        if inst.state().await != crate::backend::BackendState::Stopped {
            state.mgr.stop(&alias).await;
            return;
        }
    }
    let Some(m) = state.mgr.models.get(&alias).cloned() else { return };
    let mgr = state.mgr.clone();
    let alias_for_task = alias.clone();
    let bus = state.bus.clone();
    tokio::spawn(async move {
        if let Err(e) = mgr.ensure(&alias_for_task, &m.hf).await {
            bus.emit(&alias_for_task, &format!("load failed: {e}"));
        }
    });
}

async fn stop_selected(state: &mut AppState) {
    let Some(alias) = state.current_alias() else { return };
    if let Some(inst) = state.mgr.get(&alias).await {
        if inst.state().await != crate::backend::BackendState::Stopped {
            state.mgr.stop(&alias).await;
        }
    }
}

fn adjust_timeout(state: &mut AppState, delta: i64, label: &str) {
    let Some(alias) = state.current_alias() else { return };
    let mgr = state.mgr.clone();
    let bus = state.bus.clone();
    let label = label.to_string();
    tokio::spawn(async move {
        if let Some(inst) = mgr.get(&alias).await {
            let cur = inst.idle_timeout() as i64;
            let new = (cur + delta).max(60) as u64;
            inst.set_idle_timeout(new);
            bus.emit(
                &alias,
                &format!("idle timeout {label} → {}", fmt_age(new)),
            );
        } else {
            bus.emit(
                &alias,
                &format!("{label} timeout requested but model not loaded"),
            );
        }
    });
}

fn reset_timeout(state: &mut AppState) {
    let Some(alias) = state.current_alias() else { return };
    let mgr = state.mgr.clone();
    let bus = state.bus.clone();
    tokio::spawn(async move {
        if let Some(inst) = mgr.get(&alias).await {
            inst.set_idle_timeout(600);
            bus.emit(&alias, "idle timeout reset → 10m");
        } else {
            bus.emit(&alias, "reset requested but model not loaded");
        }
    });
}

// ----------------------------------------------------------------- draw

fn draw(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    state: &mut AppState,
) -> Result<()> {
    terminal.draw(|f| draw_ui(f, state))?;
    Ok(())
}

fn draw_ui(f: &mut Frame, state: &mut AppState) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // status
            Constraint::Length((state.aliases.len() as u16).min(14) + 3), // models
            Constraint::Min(8),     // model log (flexes)
            Constraint::Length(10), // loca log
            Constraint::Length(1),  // footer
        ])
        .split(f.area());

    draw_status(f, state, outer[0]);
    draw_models(f, state, outer[1]);
    draw_model_log(f, state, outer[2]);
    draw_loca_log(f, state, outer[3]);
    draw_footer(f, state, outer[4]);
}

fn draw_status(f: &mut Frame, state: &AppState, area: Rect) {
    let uptime = state.start_time.elapsed().as_secs();
    let ram_color = if state.tick_ram_used > state.cfg.ram_budget_gb {
        Color::Red
    } else {
        Color::Green
    };
    let plural = if state.tick_n_loaded == 1 { "" } else { "s" };
    let line1 = Line::from(vec![
        Span::styled("Proxy  ", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(format!(
            "http://{}:{}/v1   uptime {}   {} req   {} err",
            state.cfg.listen_host,
            state.cfg.listen_port,
            fmt_age(uptime),
            state.tick_total_req,
            state.tick_total_err,
        )),
    ]);
    let line2 = Line::from(vec![
        Span::styled("RAM    ", Style::default().add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("{} / {} GB", state.tick_ram_used, state.cfg.ram_budget_gb),
            Style::default().fg(ram_color),
        ),
        Span::raw(format!(
            "   {} model{} loaded",
            state.tick_n_loaded, plural
        )),
    ]);
    let p = Paragraph::new(vec![line1, line2])
        .block(Block::default().borders(Borders::BOTTOM));
    f.render_widget(p, area);
}

fn draw_models(f: &mut Frame, state: &mut AppState, area: Rect) {
    let header_style = Style::default().add_modifier(Modifier::BOLD);
    let header = Row::new(vec![
        Cell::from("alias"),
        Cell::from("size"),
        Cell::from("state"),
        Cell::from("idle"),
        Cell::from("cached"),
        Cell::from("description"),
    ])
    .style(header_style);

    let rows: Vec<Row> = state
        .tick_rows
        .iter()
        .map(|r| {
            let state_style = match r.state {
                "READY" => Style::default().fg(Color::Green),
                "LOADING" => Style::default().fg(Color::Yellow),
                _ => Style::default().fg(Color::DarkGray),
            };
            Row::new(vec![
                Cell::from(r.alias.clone()),
                Cell::from(match r.ram_gb {
                    Some(g) => format!("~{g} GB"),
                    None => "—".into(),
                }),
                Cell::from(r.state).style(state_style),
                Cell::from(r.idle.clone()),
                Cell::from(r.cached),
                Cell::from(r.description.clone()),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(20),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(18),
        Constraint::Length(7),
        Constraint::Min(20),
    ];
    let block_style = if state.focus == Pane::Models {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Models")
                .border_style(block_style),
        );
    f.render_stateful_widget(table, area, &mut state.table_state);
}

fn draw_model_log(f: &mut Frame, state: &AppState, area: Rect) {
    let (title, lines): (String, &Vec<String>) = match &state.selected_alias {
        Some(a) => {
            let lines = state
                .logs
                .get(a)
                .map(|b| &b.lines)
                .unwrap_or_else(|| empty_lines());
            (format!("Log: {a}"), lines)
        }
        None => ("Log: (no row selected)".into(), empty_lines()),
    };
    let block_style = if state.focus == Pane::ModelLog {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let visible_h = area.height.saturating_sub(2) as usize;
    let start = lines.len().saturating_sub(visible_h);
    let body: Vec<Line> = lines[start..].iter().map(|l| Line::from(l.as_str())).collect();
    let p = Paragraph::new(body)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .border_style(block_style),
        );
    f.render_widget(p, area);
}

fn draw_loca_log(f: &mut Frame, state: &AppState, area: Rect) {
    let lines = state
        .logs
        .get("loca")
        .map(|b| &b.lines)
        .unwrap_or_else(|| empty_lines());
    let block_style = if state.focus == Pane::LocaLog {
        Style::default().fg(Color::Cyan)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let visible_h = area.height.saturating_sub(2) as usize;
    let start = lines.len().saturating_sub(visible_h);
    let body: Vec<Line> = lines[start..].iter().map(|l| Line::from(l.as_str())).collect();
    let p = Paragraph::new(body)
        .wrap(Wrap { trim: false })
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Log: loca")
                .border_style(block_style),
        );
    f.render_widget(p, area);
}

fn draw_footer(f: &mut Frame, state: &AppState, area: Rect) {
    let mut spans: Vec<Span> = Vec::new();
    let sep = || Span::raw("   ");
    let hint = |k: &str, label: &str| -> Vec<Span<'static>> {
        vec![
            Span::styled(k.to_string(), Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
            Span::raw(" "),
            Span::raw(label.to_string()),
        ]
    };
    spans.extend(hint("q", "Quit"));
    spans.push(sep());
    spans.extend(hint("Tab", "Cycle pane"));
    spans.push(sep());
    match state.focus {
        Pane::Models => {
            spans.extend(hint("↑↓", "Nav"));
            spans.push(sep());
            spans.extend(hint("⏎", "Load/Stop"));
            spans.push(sep());
            spans.extend(hint("s", "Stop"));
            spans.push(sep());
            spans.extend(hint("t", "+10m"));
            spans.push(sep());
            spans.extend(hint("T", "+1h"));
            spans.push(sep());
            spans.extend(hint("r", "Reset 10m"));
        }
        Pane::ModelLog | Pane::LocaLog => {
            spans.extend(hint("c", "Clear log"));
        }
    }
    let p = Paragraph::new(Line::from(spans));
    f.render_widget(p, area);
}

fn empty_lines() -> &'static Vec<String> {
    use std::sync::OnceLock;
    static EMPTY: OnceLock<Vec<String>> = OnceLock::new();
    EMPTY.get_or_init(Vec::new)
}

fn fmt_age(sec: u64) -> String {
    if sec < 60 {
        format!("{sec}s")
    } else if sec < 3600 {
        format!("{}m {}s", sec / 60, sec % 60)
    } else {
        format!("{}h {}m", sec / 3600, (sec % 3600) / 60)
    }
}
