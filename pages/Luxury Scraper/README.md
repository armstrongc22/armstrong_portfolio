# Luxury Item Price Tracker

A sophisticated web scraping application designed to track prices of luxury items, starting with watches. The system analyzes market prices and identifies good deals across multiple platforms.

## Features

- Multi-platform scraping (eBay, 1stDibs, Wrist Aficionado, Google Shopping)
- Real-time price analysis and deal detection
- Local data storage option
- Email notifications for exceptional deals
- Modern white and gold UI
- Expandable architecture for different luxury items (watches, art, clothes, jewelry)

## Setup

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up Streamlit secrets:
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Edit `.streamlit/secrets.toml` with your credentials:
     ```toml
     [email]
     address = "your_email@gmail.com"
     password = "your_app_specific_password"  # For Gmail, use App Password
     ```
   - For deployment to Streamlit Cloud:
     1. Go to your app's settings in Streamlit Cloud
     2. Under "Secrets", add your credentials in TOML format
     3. Save the changes

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Email Notifications Setup

For Gmail users:
1. Go to your Google Account settings
2. Enable 2-Step Verification
3. Generate an App Password:
   - Go to Security settings
   - Select "App Passwords"
   - Generate a new app password for "Mail"
4. Use this App Password in your Streamlit secrets

## Architecture

- `scrapers/`: Individual scraper implementations for each platform
- `models/`: Database models and schemas
- `utils/`: Utility functions and helpers
- `static/`: UI assets and styles
- `templates/`: HTML templates
- `config/`: Configuration files
- `services/`: Business logic and email services
- `.streamlit/`: Streamlit configuration and secrets

## Contributing

This project is designed to be extensible. New luxury item categories can be added by implementing new scraper classes following the base scraper interface.

## Adding New Scrapers

To add a new scraper:
1. Create a new file in the `scrapers/` directory
2. Inherit from `BaseScraper` class
3. Implement the required methods:
   - `scrape()`
   - `get_market_value()`

Example:
```python
from scrapers.base_scraper import BaseScraper

class NewPlatformScraper(BaseScraper):
    def __init__(self, category: str = "watches"):
        super().__init__(category)
        
    async def scrape(self, search_term: str):
        # Implement scraping logic
        pass
        
    async def get_market_value(self, model: str):
        # Implement market value calculation
        pass
```

## Future Enhancements

- Support for additional luxury item categories
- Advanced price trend analysis
- User accounts and saved searches
- Mobile app integration
- API access for third-party integration

## Deployment

### Local Development
1. Clone the repository
2. Install dependencies
3. Set up secrets in `.streamlit/secrets.toml`
4. Run `streamlit run app.py`

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Add your secrets in the Streamlit Cloud dashboard
4. Deploy! 