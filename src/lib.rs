//! # Vibesort
//!
//! A Rust library for sorting arrays using Large Language Models (LLMs).
//!
//! This library provides a simple interface to sort arrays by leveraging LLM APIs
//! such as OpenAI, Anthropic, or other compatible services. It sends the array to
//! the LLM and parses the sorted result.
//!
//! ## Features
//!
//! - Sort arrays of any type that implements `Display`, `Serialize`, and `DeserializeOwned`
//! - Support for any LLM API compatible with OpenAI's chat completion format
//! - Comprehensive error handling with detailed error messages
//! - Async/await support using Tokio
//!
//! ## Example
//!
//! ```no_run
//! use vibesort_rs::Vibesort;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let sorter = Vibesort::new(
//!     "your-api-key",
//!     "gpt-3.5-turbo",
//!     "https://api.openai.com/v1",
//! );
//!
//! let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
//! let sorted = sorter.sort(&numbers).await?;
//! println!("{:?}", sorted); // [1, 1, 2, 3, 4, 5, 6, 9]
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fmt::Display;
use thiserror::Error;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vibesort_config() {
        let sorter = Vibesort::new("key", "model", "url");
        assert_eq!(sorter.api_key, "key");
        assert_eq!(sorter.model, "model");
        assert_eq!(sorter.base_url, "url");
    }

    #[tokio::test]
    async fn test_vibesort_with_mock() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Start a mock server
        let mock_server = MockServer::start().await;
        let base_url = mock_server.uri();

        // Set up a mock response
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "[1,1,2,3,4,5,6,9]"
                    }
                }]
            })))
            .mount(&mock_server)
            .await;

        // Create a Vibesort instance pointing to the mock server
        let sorter = Vibesort::new("test-api-key", "test-model", base_url.as_str());

        // Test the sorting
        let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
        let result = sorter.sort(&numbers).await;

        assert!(result.is_ok());
        let sorted = result.unwrap();
        assert_eq!(sorted, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[tokio::test]
    async fn test_vibesort_api_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Start a mock server
        let mock_server = MockServer::start().await;
        let base_url = mock_server.uri();

        // Set up a mock error response
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("Internal Server Error"))
            .mount(&mock_server)
            .await;

        let sorter = Vibesort::new("test-api-key", "test-model", base_url.as_str());

        let numbers = vec![3, 1, 4];
        let result = sorter.sort(&numbers).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            VibesortError::ApiError(_) => {}
            _ => panic!("Expected ApiError"),
        }
    }

    #[tokio::test]
    async fn test_vibesort_parse_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Start a mock server
        let mock_server = MockServer::start().await;
        let base_url = mock_server.uri();

        // Set up a mock response with invalid JSON (not a valid array)
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "Here is the sorted array: 1, 2, 3"
                    }
                }]
            })))
            .mount(&mock_server)
            .await;

        let sorter = Vibesort::new("test-api-key", "test-model", base_url.as_str());

        let numbers = vec![3, 1, 2];
        let result = sorter.sort(&numbers).await;

        assert!(result.is_err());
        match result.unwrap_err() {
            VibesortError::ParseError(msg) => {
                // Verify that the error message contains the LLM's response
                assert!(msg.contains("Here is the sorted array: 1, 2, 3"));
            }
            _ => panic!("Expected ParseError"),
        }
    }

    #[tokio::test]
    async fn test_vibesort_str_with_mock() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        // Start a mock server
        let mock_server = MockServer::start().await;
        let base_url = mock_server.uri();

        // Set up a mock response for string sorting
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "content": "[\"apple\",\"banana\",\"cherry\"]"
                    }
                }]
            })))
            .mount(&mock_server)
            .await;

        // Create a Vibesort instance pointing to the mock server
        let sorter = Vibesort::new("test-api-key", "test-model", base_url.as_str());

        // Test the string sorting
        let words = vec!["banana", "apple", "cherry"];
        let result = sorter.sort_str(&words).await;

        assert!(result.is_ok());
        let sorted = result.unwrap();
        assert_eq!(sorted, vec!["apple", "banana", "cherry"]);
    }
}

/// Error types for vibesort operations.
///
/// This enum represents all possible errors that can occur during the sorting process.
#[derive(Error, Debug)]
pub enum VibesortError {
    /// An error occurred while making the HTTP request to the LLM API.
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    /// An error occurred while parsing JSON (e.g., when serializing the input array
    /// or deserializing the LLM response).
    #[error("JSON parsing failed: {0}")]
    JsonError(#[from] serde_json::Error),

    /// The LLM API returned an error status code.
    ///
    /// This error includes the HTTP status code and the server's response body.
    #[error("LLM API error: {0}")]
    ApiError(String),

    /// The LLM API response is missing required fields or has an invalid structure.
    ///
    /// This typically means the response doesn't contain a `choices` array or
    /// the first choice doesn't have a `message` field.
    #[error("Invalid response format from LLM")]
    InvalidResponse,

    /// The LLM returned content that cannot be parsed as a JSON array.
    ///
    /// This error includes the parsing error details and the actual content
    /// returned by the LLM, which helps diagnose why the parsing failed.
    #[error("Failed to parse LLM response as sorted array. LLM returned: {0}")]
    ParseError(String),
}

/// OpenAI API request/response structures
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ChatMessageResponse,
}

/// Client for sorting arrays using LLM APIs.
///
/// This struct holds the configuration needed to communicate with an LLM API
/// and provides methods to sort arrays.
///
/// # Example
///
/// ```no_run
/// use vibesort_rs::Vibesort;
///
/// let sorter = Vibesort::new(
///     "sk-...",
///     "gpt-3.5-turbo",
///     "https://api.openai.com/v1",
/// );
/// ```
#[derive(Debug, Clone)]
pub struct Vibesort<'a> {
    /// The API key for authenticating with the LLM service.
    pub api_key: &'a str,

    /// The model identifier to use (e.g., "gpt-3.5-turbo", "gpt-4").
    pub model: &'a str,

    /// The base URL of the LLM API endpoint (e.g., "https://api.openai.com/v1").
    pub base_url: &'a str,
}

impl<'a> Vibesort<'a> {
    /// Creates a new `Vibesort` instance.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key for authenticating with the LLM service
    /// * `model` - The model identifier to use (e.g., "gpt-3.5-turbo", "gpt-4")
    /// * `base_url` - The base URL of the LLM API endpoint
    ///
    /// # Example
    ///
    /// ```no_run
    /// use vibesort_rs::Vibesort;
    ///
    /// let sorter = Vibesort::new(
    ///     "sk-1234567890abcdef",
    ///     "gpt-3.5-turbo",
    ///     "https://api.openai.com/v1",
    /// );
    /// ```
    pub fn new(api_key: &'a str, model: &'a str, base_url: &'a str) -> Self {
        Self {
            api_key,
            model,
            base_url,
        }
    }

    /// Sorts an array using an LLM.
    ///
    /// This method sends the input array to the configured LLM API and requests
    /// it to sort the elements. The LLM is instructed to return only a JSON array
    /// with the sorted elements, which is then parsed and returned.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of items to sort. Each item must implement:
    ///   - `Display` - For error messages
    ///   - `Serialize` - For serializing to JSON
    ///   - `DeserializeOwned` - For deserializing the sorted result
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<T>)` with the sorted array if successful, or an error
    /// if the API call fails, the response is invalid, or parsing fails.
    ///
    /// # Errors
    ///
    /// This method can return various errors:
    /// - [`VibesortError::HttpError`] - Network or HTTP request errors
    /// - [`VibesortError::ApiError`] - API returned an error status code
    /// - [`VibesortError::InvalidResponse`] - Response format is invalid
    /// - [`VibesortError::ParseError`] - LLM response cannot be parsed as a JSON array
    /// - [`VibesortError::JsonError`] - JSON serialization/deserialization errors
    ///
    /// # Examples
    ///
    /// ## Sorting numbers
    ///
    /// ```no_run
    /// use vibesort_rs::Vibesort;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let sorter = Vibesort::new(
    ///     "your-api-key",
    ///     "gpt-3.5-turbo",
    ///     "https://api.openai.com/v1",
    /// );
    ///
    /// let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    /// let sorted = sorter.sort(&numbers).await?;
    /// assert_eq!(sorted, vec![1, 1, 2, 3, 4, 5, 6, 9]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Sorting strings
    ///
    /// ```no_run
    /// use vibesort_rs::Vibesort;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let sorter = Vibesort::new(
    ///     "your-api-key",
    ///     "gpt-3.5-turbo",
    ///     "https://api.openai.com/v1",
    /// );
    ///
    /// let words: Vec<String> = vec!["banana", "apple", "cherry"]
    ///     .into_iter()
    ///     .map(|s| s.to_string())
    ///     .collect();
    /// let sorted = sorter.sort(&words).await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Error handling
    ///
    /// ```no_run
    /// use vibesort_rs::{Vibesort, VibesortError};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let sorter = Vibesort::new(
    ///     "invalid-key",
    ///     "gpt-3.5-turbo",
    ///     "https://api.openai.com/v1",
    /// );
    ///
    /// match sorter.sort(&vec![1, 2, 3]).await {
    ///     Ok(sorted) => println!("Sorted: {:?}", sorted),
    ///     Err(VibesortError::ApiError(msg)) => eprintln!("API error: {}", msg),
    ///     Err(e) => eprintln!("Other error: {}", e),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn sort<T>(&self, items: &[T]) -> Result<Vec<T>, VibesortError>
    where
        T: Display + Serialize + DeserializeOwned,
    {
        // Serialize the input array to JSON
        let json_array = serde_json::to_string(items)?;

        // Build the API URL
        let url = format!("{}/chat/completions", self.base_url);

        // Create the HTTP client
        let client = reqwest::Client::new();

        // Prepare the request with system prompt and user prompt
        let system_prompt = "You are a helpful assistant that sorts arrays. Sort the following JSON array with ascending order and return ONLY the sorted JSON array, nothing else.";
        let request = ChatRequest {
            model: self.model,
            messages: vec![
                ChatMessage {
                    role: "system",
                    content: system_prompt,
                },
                ChatMessage {
                    role: "user",
                    content: &json_array,
                },
            ],
            temperature: 0.0, // Use 0.0 for deterministic sorting
        };

        // Send the request
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // Check if the request was successful
        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(VibesortError::ApiError(format!(
                "API returned status {}\nServer response: {}",
                status, error_text
            )));
        }

        // Parse the response
        let chat_response: ChatResponse = response.json().await?;

        // Extract the sorted array from the LLM's response
        let sorted_json = chat_response
            .choices
            .first()
            .ok_or(VibesortError::InvalidResponse)?
            .message
            .content
            .trim();

        // Parse the JSON array back to Vec<T>
        let sorted: Vec<T> = serde_json::from_str(sorted_json).map_err(|e| {
            VibesortError::ParseError(format!(
                "Failed to parse as JSON array: {}\nLLM returned: {}",
                e, sorted_json
            ))
        })?;

        Ok(sorted)
    }

    /// Sorts an array of strings using an LLM.
    ///
    /// This is a convenience method specifically for sorting string arrays.
    /// It accepts a slice of string references and returns a vector of owned strings.
    ///
    /// # Arguments
    ///
    /// * `items` - A slice of string references to sort
    ///
    /// # Returns
    ///
    /// Returns `Ok(Vec<String>)` with the sorted array if successful, or an error
    /// if the API call fails, the response is invalid, or parsing fails.
    ///
    /// # Errors
    ///
    /// This method can return the same errors as [`sort`](Self::sort):
    /// - [`VibesortError::HttpError`] - Network or HTTP request errors
    /// - [`VibesortError::ApiError`] - API returned an error status code
    /// - [`VibesortError::InvalidResponse`] - Response format is invalid
    /// - [`VibesortError::ParseError`] - LLM response cannot be parsed as a JSON array
    /// - [`VibesortError::JsonError`] - JSON serialization/deserialization errors
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use vibesort_rs::Vibesort;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let sorter = Vibesort::new(
    ///     "your-api-key",
    ///     "gpt-3.5-turbo",
    ///     "https://api.openai.com/v1",
    /// );
    ///
    /// let words = vec!["banana", "apple", "cherry"];
    /// let sorted = sorter.sort_str(&words).await?;
    /// assert_eq!(sorted, vec!["apple", "banana", "cherry"]);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn sort_str(&self, items: &[&str]) -> Result<Vec<String>, VibesortError> {
        // Convert &[&str] to Vec<String> for serialization
        let string_vec: Vec<String> = items.iter().map(|s| s.to_string()).collect();
        self.sort(&string_vec).await
    }
}
