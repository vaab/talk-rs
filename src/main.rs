use talk_rs::cli;

#[tokio::main]
async fn main() {
    if let Err(err) = cli::run().await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
