use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::agent::meta_agent::MetaAgent;
use crate::agent::task_agent::TaskAgent;
use crate::domains::harness::run_harness;
use crate::domains::report::generate_report;
use crate::utils::common::{file_exists_and_not_empty, get_score_from_report};
use crate::utils::git::{get_base_commit, git_apply_diff, git_reset};
use crate::utils::progress::{print_evolution_tree, print_progress_graph};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ArchiveEntry {
    pub id: String,
    pub parent: Option<String>,
    pub score: Option<f64>,
    pub gen: usize,
    pub patch_file: Option<String>,
    pub meta_success: bool,
}

pub struct Config {
    pub domain: String,
    pub model: String,
    pub max_generation: usize,
    pub num_samples: i64,
    pub num_workers: usize,
    pub output_dir: Option<PathBuf>,
    pub parent_selection: String,
    pub verbose: bool,
}

/// Select which archive entry to use as parent for the next generation.
pub fn select_parent(archive: &[ArchiveEntry], method: &str) -> String {
    let valid: Vec<&ArchiveEntry> = archive.iter().filter(|e| e.score.is_some()).collect();
    if valid.is_empty() {
        return archive
            .last()
            .map(|e| e.id.clone())
            .unwrap_or_else(|| "initial".to_string());
    }
    match method {
        "best" => valid
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .map(|e| e.id.clone())
            .unwrap_or_else(|| "initial".to_string()),
        "latest" => valid
            .last()
            .map(|e| e.id.clone())
            .unwrap_or_else(|| "initial".to_string()),
        _ => {
            // Proportional / random
            let scores: Vec<f64> = valid.iter().map(|e| e.score.unwrap().max(0.01)).collect();
            let total: f64 = scores.iter().sum();
            let mut rng_val = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as f64)
                / 1_000_000_000.0
                * total;
            for (entry, score) in valid.iter().zip(scores.iter()) {
                rng_val -= score;
                if rng_val <= 0.0 {
                    return entry.id.clone();
                }
            }
            valid
                .last()
                .map(|e| e.id.clone())
                .unwrap_or_else(|| "initial".to_string())
        }
    }
}

fn format_score(score: Option<f64>) -> String {
    match score {
        Some(s) => format!("{:.4}", s),
        None => "N/A".to_string(),
    }
}

/// Run the full local generation loop.
pub fn run(config: Config) -> Result<PathBuf> {
    // Load .env if present
    let _ = dotenv::dotenv();

    let model = config.model.clone();
    std::env::set_var("MODEL_NAME", &model);

    let run_id = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir_parent = config
        .output_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap().join("outputs_local"));
    let output_dir = output_dir_parent.join(format!("run_{}", run_id));
    std::fs::create_dir_all(&output_dir)?;

    let project_dir = std::env::current_dir()?;
    let base_commit = get_base_commit(&project_dir)?;

    println!("{}", "=".repeat(60));
    println!("HyperAgents Local Loop (Rust)");
    println!("  Model:      {}", model);
    println!("  Domain:     {}", config.domain);
    println!("  Gens:       {}", config.max_generation);
    println!("  Output:     {}", output_dir.display());
    println!("  Base commit:{}", &base_commit[..8.min(base_commit.len())]);
    println!("{}", "=".repeat(60));

    let mut archive: Vec<ArchiveEntry> = Vec::new();
    let archive_file = output_dir.join("archive.jsonl");

    // Determine subset
    let subset = if config.domain == "text_classify"
        || config.domain == "emotion"
        || config.domain == "factory"
    {
        "_train"
    } else {
        ""
    };

    // --- Initial evaluation ---
    let initial_score = run_initial_eval(
        &project_dir,
        &config.domain,
        &model,
        &output_dir.join("gen_initial"),
        config.num_samples,
        subset,
        config.num_workers,
    )?;

    let initial_entry = ArchiveEntry {
        id: "initial".to_string(),
        parent: None,
        score: initial_score,
        gen: 0,
        patch_file: None,
        meta_success: true,
    };
    archive.push(initial_entry.clone());
    append_archive(&archive_file, &initial_entry)?;

    // --- Generation loop ---
    let loop_start = std::time::Instant::now();

    for gen_id in 1..=config.max_generation {
        println!("\n{}", "=".repeat(60));
        println!("Generation {}/{}", gen_id, config.max_generation);

        if config.verbose {
            let score_history: Vec<String> = archive
                .iter()
                .filter(|e| e.score.is_some())
                .map(|e| {
                    let label = if e.id == "initial" {
                        "initial".to_string()
                    } else {
                        format!("gen_{}", e.id)
                    };
                    format!("{}: {}", label, format_score(e.score))
                })
                .collect();
            if !score_history.is_empty() {
                println!("  Score history: {}", score_history.join(" → "));
            }
        }
        let best_so_far = archive
            .iter()
            .filter_map(|e| e.score)
            .fold(0.0_f64, f64::max);
        println!("  Best so far:  {}", format_score(Some(best_so_far)));
        println!("{}", "=".repeat(60));

        let parent_id = select_parent(&archive, &config.parent_selection);
        let parent_entry = archive.iter().find(|e| e.id == parent_id).cloned();
        let parent_score = parent_entry.as_ref().and_then(|e| e.score);
        println!(
            "  Parent: {} (score: {})",
            parent_id,
            format_score(parent_score)
        );

        let gen_output_dir = output_dir.join(format!("gen_{}", gen_id));
        std::fs::create_dir_all(&gen_output_dir)?;

        // Reset to base commit
        git_reset(&project_dir, &base_commit)?;

        // Apply parent patch if it has one
        let parent_patch_ok = apply_parent_patch(&project_dir, parent_entry.as_ref())?;
        if !parent_patch_ok {
            println!(
                "  Parent patch failed to apply, skipping generation {}.",
                gen_id
            );
            let entry = ArchiveEntry {
                id: gen_id.to_string(),
                parent: Some(parent_id.clone()),
                score: None,
                gen: gen_id,
                patch_file: None,
                meta_success: false,
            };
            archive.push(entry.clone());
            append_archive(&archive_file, &entry)?;
            continue;
        }

        // Run meta agent
        let evals_folder = output_dir.clone();
        let (success, patch_file) = run_meta_agent(
            &project_dir,
            &model,
            &gen_output_dir,
            &base_commit,
            &evals_folder,
            config.max_generation - gen_id,
            &config.domain,
        )?;

        let mut score: Option<f64> = None;
        let mut actual_patch_file: Option<String> = None;

        if success {
            // Re-apply parent patch after reset
            git_reset(&project_dir, &base_commit)?;
            let re_apply_ok = apply_parent_patch(&project_dir, parent_entry.as_ref())?;
            if !re_apply_ok {
                println!("  Parent patch failed to re-apply after reset, skipping evaluation.");
            } else {
                // Apply new patch
                let patch_applied = if let Some(ref pf) = patch_file {
                    let pf_path = PathBuf::from(pf);
                    git_apply_diff(&project_dir, &pf_path)?
                } else {
                    // For agent_prompt.txt approach: copy the saved prompt back
                    true
                };

                if patch_applied {
                    // Re-copy the prompt file from gen output dir if it was saved there
                    restore_agent_prompt(&gen_output_dir, &project_dir);

                    score = run_eval(
                        &project_dir,
                        &config.domain,
                        &model,
                        &gen_output_dir,
                        gen_id,
                        config.num_samples,
                        subset,
                        config.num_workers,
                    )?;
                    actual_patch_file = patch_file.clone();
                } else {
                    println!("  Generated patch failed to apply cleanly, skipping evaluation.");
                }
            }
        } else {
            println!("  Meta agent failed, skipping evaluation.");
        }

        let entry = ArchiveEntry {
            id: gen_id.to_string(),
            parent: Some(parent_id.clone()),
            score,
            gen: gen_id,
            patch_file: actual_patch_file,
            meta_success: success,
        };
        archive.push(entry.clone());
        append_archive(&archive_file, &entry)?;

        // Save per-gen metadata
        let metadata = serde_json::json!({
            "gen_id": gen_id,
            "parent_id": parent_id,
            "score": score,
            "model": model,
            "domain": config.domain,
            "meta_success": success,
        });
        std::fs::write(
            gen_output_dir.join("metadata.json"),
            serde_json::to_string_pretty(&metadata)?,
        )?;

        // Delta indicator
        let indicator = match (score, parent_score) {
            (Some(s), Some(p)) => {
                let delta = s - p;
                let arrow = if delta > 0.0 {
                    "^"
                } else if delta < 0.0 {
                    "v"
                } else {
                    "="
                };
                format!("{} {:+.4}", arrow, delta)
            }
            _ => "N/A".to_string(),
        };
        println!(
            "  Gen {} done — score: {}, parent: {}, change: {}",
            gen_id,
            format_score(score),
            parent_id,
            indicator
        );

        if config.verbose {
            print_progress_graph(&archive);
        }
    }

    // Final reset
    git_reset(&project_dir, &base_commit)?;

    // Summary
    println!("\n{}", "=".repeat(60));
    println!("RESULTS SUMMARY");
    println!("{}", "=".repeat(60));

    let best_score = archive
        .iter()
        .filter_map(|e| e.score)
        .fold(0.0_f64, f64::max);
    for entry in &archive {
        let marker = if entry.score == Some(best_score) {
            " *"
        } else {
            ""
        };
        let parent_str = entry.parent.as_deref().unwrap_or("-");
        println!(
            "  Gen {:>8} | Score: {:>8} | Parent: {:>8}{}",
            entry.id,
            format_score(entry.score),
            parent_str,
            marker
        );
    }

    if config.verbose {
        print_evolution_tree(&archive);
        print_progress_graph(&archive);
    }

    // Export best agent
    let best = archive
        .iter()
        .filter(|e| e.score.is_some())
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap());

    if let Some(best) = best {
        println!(
            "\n  Best: Gen {} with score {:.3}",
            best.id,
            best.score.unwrap()
        );

        git_reset(&project_dir, &base_commit)?;
        if best.id != "initial" {
            if let Some(ref pf) = best.patch_file {
                let pf_path = PathBuf::from(pf);
                let _ = git_apply_diff(&project_dir, &pf_path);
            }
            // Restore best agent prompt from that gen's output
            let best_gen_dir = output_dir.join(format!("gen_{}", best.id));
            restore_agent_prompt(&best_gen_dir, &project_dir);
        }

        let best_prompt_src = project_dir.join("agent_prompt.txt");
        let best_prompt_dst = output_dir.join("best_agent_prompt.txt");
        if best_prompt_src.exists() {
            std::fs::copy(&best_prompt_src, &best_prompt_dst)?;
            println!(
                "  Best agent prompt exported to: {}",
                best_prompt_dst.display()
            );
        }

        git_reset(&project_dir, &base_commit)?;
    }

    let elapsed = loop_start.elapsed();
    println!(
        "\n  Total time: {:.1} minutes",
        elapsed.as_secs_f64() / 60.0
    );
    println!("  Output saved to: {}", output_dir.display());

    Ok(output_dir)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn run_initial_eval(
    _project_dir: &Path,
    domain: &str,
    model: &str,
    output_dir: &Path,
    num_samples: i64,
    subset: &str,
    num_workers: usize,
) -> Result<Option<f64>> {
    println!("\n{}", "=".repeat(60));
    println!("Running initial evaluation on {} ({})…", domain, subset);
    println!("{}", "=".repeat(60));

    std::fs::create_dir_all(output_dir)?;
    let agent = TaskAgent::new(model);
    let run_id = format!("initial_{}{}_0", domain, subset);

    let eval_dir = run_harness(
        &agent,
        domain,
        output_dir.parent().unwrap_or(output_dir),
        &run_id,
        num_samples,
        subset,
        num_workers,
    )?;
    // The eval_dir IS already output_dir/run_id but run_harness puts it under output_dir arg.
    // Actually we passed output_dir.parent() so eval_dir = output_dir.parent()/run_id
    // Let's just move it: check if results landed in the right place.
    let report_path = eval_dir.join("report.json");
    if !report_path.exists() {
        let _ = generate_report(&eval_dir, domain);
    }

    let score = get_score_from_report(&eval_dir.join("report.json"));
    println!("  Initial score: {}", format_score(score));
    Ok(score)
}

fn run_eval(
    _project_dir: &Path,
    domain: &str,
    model: &str,
    output_dir: &Path,
    gen_id: usize,
    num_samples: i64,
    subset: &str,
    num_workers: usize,
) -> Result<Option<f64>> {
    println!("  Evaluating generation {}…", gen_id);
    let agent = TaskAgent::new(model);
    let run_id = format!("{}_eval", domain);
    let eval_dir = run_harness(
        &agent,
        domain,
        output_dir,
        &run_id,
        num_samples,
        subset,
        num_workers,
    )?;
    let _ = generate_report(&eval_dir, domain);
    let score = get_score_from_report(&eval_dir.join("report.json"));
    println!("  Score for gen {}: {}", gen_id, format_score(score));
    Ok(score)
}

/// Run the meta agent for one generation. Returns (success, patch_file_path).
/// In the Rust port the "patch" is the new agent_prompt.txt saved to gen_output_dir.
fn run_meta_agent(
    project_dir: &Path,
    model: &str,
    gen_output_dir: &Path,
    _base_commit: &str,
    evals_folder: &Path,
    iterations_left: usize,
    domain: &str,
) -> Result<(bool, Option<String>)> {
    println!("\n  Running meta agent (model={})…", model);
    let start = std::time::Instant::now();

    let agent_output_dir = gen_output_dir.join("agent_output");
    std::fs::create_dir_all(&agent_output_dir)?;
    let chat_history_file = agent_output_dir.join("meta_agent_chat_history.md");

    let meta = MetaAgent::new(model, chat_history_file);

    match meta.forward(project_dir, evals_folder, iterations_left, domain) {
        Ok(()) => {
            // Save the new agent_prompt.txt to the gen directory for later restore
            let src = project_dir.join("agent_prompt.txt");
            let dst = gen_output_dir.join("agent_prompt.txt");
            if src.exists() {
                std::fs::copy(&src, &dst)?;
            }

            let elapsed = start.elapsed();
            println!("  Meta agent SUCCEEDED ({:.1}s)", elapsed.as_secs_f64());

            // The "patch file" is the saved agent_prompt.txt path
            let patch_path = dst.to_string_lossy().to_string();
            Ok((true, Some(patch_path)))
        }
        Err(e) => {
            let elapsed = start.elapsed();
            eprintln!("  Meta agent FAILED ({:.1}s): {}", elapsed.as_secs_f64(), e);
            Ok((false, None))
        }
    }
}

/// If the parent entry has a patch_file pointing to an agent_prompt.txt, copy it.
fn apply_parent_patch(project_dir: &Path, parent_entry: Option<&ArchiveEntry>) -> Result<bool> {
    if let Some(entry) = parent_entry {
        if let Some(ref pf) = entry.patch_file {
            let pf_path = PathBuf::from(pf);
            // If it's an agent_prompt.txt file, just copy it
            if pf_path
                .file_name()
                .map(|n| n == "agent_prompt.txt")
                .unwrap_or(false)
            {
                if pf_path.exists() {
                    let dst = project_dir.join("agent_prompt.txt");
                    std::fs::copy(&pf_path, &dst)?;
                    return Ok(true);
                }
                // File doesn't exist (initial had no prompt file)
                return Ok(true);
            }
            // Otherwise try git apply
            return git_apply_diff(project_dir, &pf_path);
        }
    }
    Ok(true) // No patch needed (initial generation)
}

/// Copy agent_prompt.txt from gen_output_dir back to project_dir if it exists
fn restore_agent_prompt(gen_output_dir: &Path, project_dir: &Path) {
    let src = gen_output_dir.join("agent_prompt.txt");
    if src.exists() {
        let dst = project_dir.join("agent_prompt.txt");
        let _ = std::fs::copy(&src, &dst);
    }
}

fn append_archive(archive_file: &Path, entry: &ArchiveEntry) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(archive_file)?;
    writeln!(file, "{}", serde_json::to_string(entry)?)?;
    Ok(())
}

#[allow(dead_code)]
fn file_exists_non_empty(path: &Path) -> bool {
    file_exists_and_not_empty(path)
}
