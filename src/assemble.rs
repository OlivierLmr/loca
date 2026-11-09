//! Rebuild a non-streaming chat completion from a streamed one.
//!
//! A non-streaming request gives no progress until the whole answer is done,
//! so the proxy asks mlx for a stream instead (to show live stats) and
//! reassembles the deltas into the `chat.completion` object the client
//! asked for.

use serde_json::{json, Map, Value};

#[derive(Default)]
pub struct ChatAssembler {
    id: Option<Value>,
    created: Option<Value>,
    model: Option<Value>,
    system_fingerprint: Option<Value>,
    content: String,
    reasoning: String,
    tool_calls: Vec<Value>,
    finish_reason: Option<Value>,
    usage: Option<Value>,
}

impl ChatAssembler {
    /// Feeds whole SSE events (as forwarded by `SseTap`).
    pub fn feed(&mut self, events: &[u8]) {
        let Ok(text) = std::str::from_utf8(events) else { return };
        for line in text.lines() {
            let Some(data) = line.strip_prefix("data:") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(data.trim()) else { continue };
            self.chunk(&v);
        }
    }

    fn chunk(&mut self, v: &Value) {
        for (slot, key) in [
            (&mut self.id, "id"),
            (&mut self.created, "created"),
            (&mut self.model, "model"),
            (&mut self.system_fingerprint, "system_fingerprint"),
        ] {
            if slot.is_none() && !v[key].is_null() {
                *slot = Some(v[key].clone());
            }
        }
        if v["usage"].is_object() {
            self.usage = Some(v["usage"].clone());
        }
        let choice = &v["choices"][0];
        if !choice["finish_reason"].is_null() {
            self.finish_reason = Some(choice["finish_reason"].clone());
        }
        let delta = &choice["delta"];
        if let Some(s) = delta["content"].as_str() {
            self.content.push_str(s);
        }
        for key in ["reasoning", "reasoning_content"] {
            if let Some(s) = delta[key].as_str() {
                self.reasoning.push_str(s);
            }
        }
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            self.tool_call(tc);
        }
    }

    /// Merges a tool-call delta: OpenAI-style streams send the arguments in
    /// pieces under the same `index`.
    fn tool_call(&mut self, tc: &Value) {
        let index = tc["index"].as_u64();
        let existing = index.and_then(|i| {
            self.tool_calls
                .iter_mut()
                .find(|c| c["index"].as_u64() == Some(i))
        });
        match existing {
            Some(c) => {
                if let Some(args) = tc["function"]["arguments"].as_str() {
                    let prev = c["function"]["arguments"].as_str().unwrap_or("").to_string();
                    c["function"]["arguments"] = Value::String(prev + args);
                }
                for key in ["id", "type"] {
                    if c[key].is_null() && !tc[key].is_null() {
                        c[key] = tc[key].clone();
                    }
                }
                if c["function"]["name"].is_null() && !tc["function"]["name"].is_null() {
                    c["function"]["name"] = tc["function"]["name"].clone();
                }
            }
            None => self.tool_calls.push(tc.clone()),
        }
    }

    pub fn finish(self) -> Value {
        let mut message = Map::new();
        message.insert("role".into(), json!("assistant"));
        message.insert("content".into(), json!(self.content));
        if !self.reasoning.is_empty() {
            message.insert("reasoning".into(), json!(self.reasoning));
        }
        if !self.tool_calls.is_empty() {
            let calls = self
                .tool_calls
                .into_iter()
                .map(|mut c| {
                    if let Some(o) = c.as_object_mut() {
                        o.remove("index");
                    }
                    c
                })
                .collect::<Vec<_>>();
            message.insert("tool_calls".into(), Value::Array(calls));
        }
        let mut out = json!({
            "id": self.id.unwrap_or(Value::Null),
            "object": "chat.completion",
            "created": self.created.unwrap_or(Value::Null),
            "model": self.model.unwrap_or(Value::Null),
            "choices": [{
                "index": 0,
                "message": Value::Object(message),
                "finish_reason": self.finish_reason.unwrap_or(Value::Null),
            }],
        });
        if let Some(fp) = self.system_fingerprint {
            out["system_fingerprint"] = fp;
        }
        if let Some(u) = self.usage {
            out["usage"] = u;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuilds_message_reasoning_and_usage() {
        let mut a = ChatAssembler::default();
        a.feed(b": keepalive 0/10\n\n");
        a.feed(b"data: {\"id\":\"c1\",\"created\":5,\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning\":\"hm\"},\"finish_reason\":null}]}\n\n");
        a.feed(b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"{\\\"a\\\":\"},\"finish_reason\":null}]}\n\n");
        a.feed(b"data: {\"id\":\"c1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"1}\"},\"finish_reason\":\"stop\"}]}\n\n");
        a.feed(b"data: {\"id\":\"c1\",\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":3}}\n\ndata: [DONE]\n\n");
        let v = a.finish();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["id"], "c1");
        assert_eq!(v["model"], "m");
        assert_eq!(v["choices"][0]["message"]["content"], "{\"a\":1}");
        assert_eq!(v["choices"][0]["message"]["reasoning"], "hm");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["completion_tokens"], 3);
    }

    #[test]
    fn merges_tool_call_argument_pieces() {
        let mut a = ChatAssembler::default();
        a.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"t\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"x\\\"\"}}]}}]}\n\n");
        a.feed(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n");
        let v = a.finish();
        let call = &v["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["id"], "t");
        assert_eq!(call["function"]["name"], "f");
        assert_eq!(call["function"]["arguments"], "{\"x\":1}");
        assert!(call.get("index").is_none());
    }
}
