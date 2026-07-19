mod comms;
mod llm;

use clap::Parser;
use comms::runner::{run, CommsConfig};
use dotenv::dotenv;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "hyperagents-comms",
    about = "Agent-to-Agent Communication Loop with Overseer — Rust",
    version = "0.1.0",
    after_help = "\
Examples:
  hyperagents-comms --task relay --model ollama/llama3.2
  hyperagents-comms --task collaborate --rounds 3
  hyperagents-comms --task protocol --rounds 5
  hyperagents-comms --task free --topic \"trade-offs in distributed systems\"
  hyperagents-comms --task language --scenario 0
  hyperagents-comms --task language --scenario 1 --model llamacpp/local
  hyperagents-comms --task relay --agent-model ollama/llama3.2 --overseer-model openrouter/google/gemma-3-4b-it:free"
)]
struct Cli {
    /// Communication task
    #[arg(long, default_value = "relay",
          value_parser = ["relay", "collaborate", "protocol", "free", "language"])]
    task: String,

    /// Model for all three agents (A, B, overseer) — override individually below
    #[arg(long, default_value = "ollama/llama3.2")]
    model: String,

    /// Model for Agent A and B only (overrides --model)
    #[arg(long)]
    agent_model: Option<String>,

    /// Model for the Overseer only (overrides --model)
    #[arg(long)]
    overseer_model: Option<String>,

    /// Number of rounds
    #[arg(long, default_value_t = 4)]
    rounds: usize,

    /// Exchanges per round
    #[arg(long, default_value_t = 3)]
    exchanges: usize,

    /// Scenario index (0-based) for relay/collaborate tasks
    #[arg(long, default_value_t = 0)]
    scenario: usize,

    /// Discussion topic (for --task free)
    #[arg(long)]
    topic: Option<String>,

    /// Output directory
    #[arg(long, default_value = "./outputs_comms")]
    output_dir: PathBuf,
}

fn main() {
    dotenv().ok();
    let cli = Cli::parse();

    let output_dir = std::env::var_os("COMMS_OUTPUT_DIR")
        .map(PathBuf::from)
        .unwrap_or(cli.output_dir);

    let agent_model    = cli.agent_model.unwrap_or_else(|| cli.model.clone());
    let overseer_model = cli.overseer_model.unwrap_or_else(|| cli.model.clone());

    let cfg = CommsConfig {
        task: cli.task,
        agent_model,
        overseer_model,
        rounds: cli.rounds,
        exchanges: cli.exchanges,
        scenario_idx: cli.scenario,
        topic: cli.topic,
        output_dir,
    };

    match run(cfg) {
        Ok(out) => println!("Done. Output: {}", out.display()),
        Err(e) => {
            eprintln!("Error: {e:#}");
            std::process::exit(1);
        }
    }
}
