mod agent;
mod domains;
mod llm;
mod runner;
mod tools;
mod utils;

use clap::Parser;
use runner::{Config, run};

#[derive(Parser, Debug)]
#[command(
    name = "hyperagents",
    about = "HyperAgents Local Loop — Rust port",
    version = "0.1.0"
)]
struct Cli {
    /// Domain to optimise
    #[arg(long, default_value = "text_classify",
          value_parser = ["text_classify", "emotion", "factory"])]
    domain: String,

    /// Model to use (e.g. ollama/llama3.2)
    #[arg(long, default_value = "ollama/llama3.2")]
    model: String,

    /// Number of evolution generations
    #[arg(long, default_value_t = 5)]
    max_generation: usize,

    /// Number of samples to evaluate (-1 for all)
    #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
    num_samples: i64,

    /// Number of parallel evaluation workers
    #[arg(long, default_value_t = 4)]
    num_workers: usize,

    /// Output directory (default: ./outputs_local)
    #[arg(long)]
    output_dir: Option<std::path::PathBuf>,

    /// Parent selection strategy
    #[arg(long, default_value = "best",
          value_parser = ["best", "latest", "proportional"])]
    parent_selection: String,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let cli = Cli::parse();

    let output_dir = cli
        .output_dir
        .or_else(|| std::env::var_os("OUTPUT_DIR").map(std::path::PathBuf::from));

    let config = Config {
        domain: cli.domain,
        model: cli.model,
        max_generation: cli.max_generation,
        num_samples: cli.num_samples,
        num_workers: cli.num_workers,
        output_dir,
        parent_selection: cli.parent_selection,
        verbose: cli.verbose,
    };

    match run(config) {
        Ok(output_dir) => println!("Done. Output: {}", output_dir.display()),
        Err(e) => {
            eprintln!("Error: {:#}", e);
            std::process::exit(1);
        }
    }
}
