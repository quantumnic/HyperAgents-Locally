use anyhow::Result;
use regex::Regex;
use std::path::Path;

pub fn extract_json_blocks(text: &str) -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    for block in extract_wrapped_blocks(text, "<json>", "</json>")
        .into_iter()
        .chain(extract_wrapped_blocks(text, "```json", "```"))
    {
        if let Some(value) = parse_json_candidate(&block) {
            results.push(value);
        }
    }

    results
}

fn extract_wrapped_blocks(text: &str, open: &str, close: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut start_offset = 0;

    while let Some(open_idx_rel) = text[start_offset..].find(open) {
        let open_idx = start_offset + open_idx_rel;
        let after_open = &text[open_idx + open.len()..];

        if let Some(close_idx) = after_open.find(close) {
            blocks.push(after_open[..close_idx].trim().to_string());
            start_offset = open_idx + open.len() + close_idx + close.len();
        } else {
            break;
        }
    }

    blocks
}

fn parse_json_candidate(candidate: &str) -> Option<serde_json::Value> {
    parse_json_text(candidate).or_else(|| {
        find_balanced_json_fragment(candidate).and_then(|fragment| parse_json_text(fragment))
    })
}

fn parse_json_text(text: &str) -> Option<serde_json::Value> {
    serde_json::from_str(text).ok().or_else(|| {
        let normalized = strip_trailing_commas(text);
        serde_json::from_str(&normalized).ok()
    })
}

fn strip_trailing_commas(text: &str) -> String {
    Regex::new(r",\s*([}\]])")
        .map(|re| re.replace_all(text, "$1").into_owned())
        .unwrap_or_else(|_| text.to_string())
}

fn find_balanced_json_fragment(text: &str) -> Option<&str> {
    let start = text
        .char_indices()
        .find(|(_, ch)| matches!(ch, '{' | '['))?
        .0;

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escape_next = false;

    for (offset, ch) in text[start..].char_indices() {
        let idx = start + offset;

        if escape_next {
            escape_next = false;
            continue;
        }

        if in_string {
            match ch {
                '\\' => escape_next = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(&text[start..=idx]);
                }
            }
            _ => {}
        }
    }

    None
}

pub fn file_exists_and_not_empty(path: &Path) -> bool {
    std::fs::metadata(path)
        .map(|m| m.len() > 0)
        .unwrap_or(false)
}

pub fn read_file(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

pub fn get_score_from_report(report_path: &Path) -> Option<f64> {
    let text = std::fs::read_to_string(report_path).ok()?;
    let val: serde_json::Value = serde_json::from_str(&text).ok()?;
    val.get("overall_accuracy")?.as_f64()
}

#[cfg(test)]
mod tests {
    use super::extract_json_blocks;

    #[test]
    fn extracts_tagged_json_with_trailing_commas() {
        let text = r#"
            Some text before.
            <json>
            {
                "tool_name": "bash",
                "tool_input": {
                    "command": "echo hello",
                },
            }
            </json>
        "#;

        let values = extract_json_blocks(text);
        assert_eq!(values.len(), 1);
        assert_eq!(values[0]["tool_name"], "bash");
        assert_eq!(values[0]["tool_input"]["command"], "echo hello");
    }

    #[test]
    fn extracts_fenced_json_with_surrounding_prose() {
        let text = r##"
            Here is the object:

            ```json
            {
              "tool_name": "edit",
              "tool_input": {
                "path": "notes.md",
                "content": "# Updated"
              }
            }
            ```

            Thanks.
        "##;

        let values = extract_json_blocks(text);
        assert_eq!(values.len(), 1);
        assert_eq!(values[0]["tool_name"], "edit");
        assert_eq!(values[0]["tool_input"]["path"], "notes.md");
    }

    #[test]
    fn extracts_multiple_blocks() {
        let text = r#"
            <json>{"id": 1}</json>
            and then
            ```json
            {"id": 2}
            ```
        "#;

        let values = extract_json_blocks(text);
        assert_eq!(values.len(), 2);
        assert_eq!(values[0]["id"], 1);
        assert_eq!(values[1]["id"], 2);
    }
}
