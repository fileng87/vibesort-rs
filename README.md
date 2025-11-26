# vibesort-rs

Sort arrays using Large Language Models (LLMs).

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
vibesort-rs = "0.1.0"
tokio = { version = "1.48", features = ["rt", "macros"] }
```

## Usage

### Sorting Numbers

```rust
use vibesort_rs::Vibesort;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sorter = Vibesort::new(
        "your-api-key",
        "gpt-5",
        "https://api.openai.com/v1",
    );

    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let sorted = sorter.sort(&numbers).await?;
    println!("{:?}", sorted); // [1, 1, 2, 3, 4, 5, 6, 9]

    Ok(())
}
```

### Sorting Strings

For sorting string arrays, you can use the convenient `sort_str` method:

```rust
use vibesort_rs::Vibesort;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sorter = Vibesort::new(
        "your-api-key",
        "gpt-5",
        "https://api.openai.com/v1",
    );

    let words = vec!["banana", "apple", "cherry"];
    let sorted = sorter.sort_str(&words).await?;
    println!("{:?}", sorted); // ["apple", "banana", "cherry"]

    Ok(())
}
```

## Requirements

- An API key for an LLM service (OpenAI, Anthropic, or any compatible API)
- The API must support OpenAI's chat completion format

## Error Handling

```rust
use vibesort_rs::{Vibesort, VibesortError};

match sorter.sort(&numbers).await {
    Ok(sorted) => println!("Sorted: {:?}", sorted),
    Err(VibesortError::ApiError(msg)) => eprintln!("API error: {}", msg),
    Err(e) => eprintln!("Error: {}", e),
}
```

## License

This project is licensed under the `MIT` License. See the [LICENSE](LICENSE) file for details.
