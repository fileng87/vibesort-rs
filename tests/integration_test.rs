use vibesort_rs::Vibesort;

#[tokio::test]
#[ignore] // Requires API key, skip in regular test runs
async fn test_vibesort_numbers() {
    // Load environment variables from .env file
    dotenvy::dotenv().ok();

    let api_key = std::env::var("API_KEY").expect("API_KEY not found in environment variables");
    let model = std::env::var("MODEL").unwrap_or_else(|_| "gemini-2.5-flash-lite".to_string());
    let base_url = std::env::var("BASE_URL")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com/v1beta/openai".to_string());

    let sorter = Vibesort::new(&api_key, &model, &base_url);
    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let result = sorter.sort(&numbers).await;
    assert!(result.is_ok());

    let sorted = result.unwrap();
    assert_eq!(sorted, vec![1, 1, 2, 3, 4, 5, 6, 9]);
}
