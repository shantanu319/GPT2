//! Tiny CLI wrapper around the chat server. Spawns the Python process once,
//! keeps the model warm, and speaks a line-oriented JSON protocol on stdin/stdout.
//!
//! Run from repo root:
//!   cargo run --manifest-path inference/chat/Cargo.toml --release -- \
//!     --checkpoint saved/model/ckpt_final.pt \
//!     --data-dir data_cache/cosmopedia
//!
//! Slash commands at the prompt:
//!   /reset   -- clear accumulated token context
//!   /quit    -- exit

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(about = "Chat with a trained MyOwnTransformer checkpoint.")]
struct Args {
    #[arg(long)]
    checkpoint: String,

    #[arg(long, default_value = "data_cache/cosmopedia")]
    data_dir: String,

    #[arg(long, default_value = "inference.chat_server")]
    server: String,

    #[arg(long, default_value_t = 100)]
    max_tokens: usize,

    #[arg(long, default_value_t = 0.5)]
    temperature: f32,

    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    #[arg(long, default_value_t = 512)]
    max_context: usize,

    #[arg(long)]
    no_cuda: bool,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Request<'a> {
    Prompt { prompt: &'a str },
    Reset,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Response {
    Ready,
    Response { text: String },
    ResetOk,
    Error { error: String },
}

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl Server {
    fn spawn(args: &Args) -> Result<Self> {
        let mut cmd = Command::new("python3");
        cmd.arg("-u").arg("-m").arg(&args.server)
            .arg("--checkpoint").arg(&args.checkpoint)
            .arg("--data-dir").arg(&args.data_dir)
            .arg("--max-tokens").arg(args.max_tokens.to_string())
            .arg("--temperature").arg(args.temperature.to_string())
            .arg("--top-p").arg(args.top_p.to_string())
            .arg("--max-context").arg(args.max_context.to_string());
        if args.no_cuda {
            cmd.arg("--no-cuda");
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("failed to spawn python3 — is it on PATH?")?;

        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no child stdin"))?;
        let stdout = BufReader::new(
            child.stdout.take().ok_or_else(|| anyhow!("no child stdout"))?,
        );
        Ok(Server { child, stdin, stdout })
    }

    fn send(&mut self, req: Request<'_>) -> Result<()> {
        let line = serde_json::to_string(&req)?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn recv(&mut self) -> Result<Response> {
        let mut line = String::new();
        let n = self.stdout.read_line(&mut line)?;
        if n == 0 {
            return Err(anyhow!("python server closed stdout"));
        }
        serde_json::from_str(&line)
            .with_context(|| format!("could not parse server message: {line:?}"))
    }

    fn shutdown(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!("Launching inference.chat_server...");
    let mut server = Server::spawn(&args)?;

    match server.recv()? {
        Response::Ready => {}
        Response::Error { error } => {
            server.shutdown();
            return Err(anyhow!("server error on startup: {error}"));
        }
        other => {
            server.shutdown();
            return Err(anyhow!("expected 'ready', got {other:?}"));
        }
    }
    println!("Model loaded. Type a prompt. Commands: /reset, /quit");

    let mut rl = DefaultEditor::new()?;
    loop {
        let line = match rl.readline("> ") {
            Ok(s) => s,
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => break,
            Err(e) => return Err(e.into()),
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(trimmed);

        match trimmed {
            "/quit" | "/exit" => break,
            "/reset" => {
                server.send(Request::Reset)?;
                match server.recv()? {
                    Response::ResetOk => println!("(context cleared)"),
                    Response::Error { error } => eprintln!("error: {error}"),
                    other => eprintln!("unexpected: {other:?}"),
                }
            }
            prompt => {
                server.send(Request::Prompt { prompt })?;
                match server.recv()? {
                    Response::Response { text } => println!("{text}"),
                    Response::Error { error } => eprintln!("error: {error}"),
                    other => eprintln!("unexpected: {other:?}"),
                }
            }
        }
    }

    server.shutdown();
    Ok(())
}
