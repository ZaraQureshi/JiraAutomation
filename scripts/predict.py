from src.service import JiraMLService

def main():
    service = JiraMLService()
    service.initialize()
    
    result = service.analyze(
        "Login API Timeout",
        "Getting 504 Gateway errors on prod"
    )
    
    print(f"Predicted Priority: {result['predicted_priority']}")
    print(f"Recommended Dev: {result['recommended_dev']}")
    print(f"Dev Mood: {result['dev_mood']}")

if __name__ == "__main__":
    main()