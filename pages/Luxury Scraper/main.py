from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
from typing import List, Optional
import asyncio

from scrapers.ebay_scraper import EbayScraper
from services.email_service import EmailService
from models.database import Item, MarketValue, PriceHistory
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

app = FastAPI(title="Luxury Item Price Tracker")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Database
engine = create_engine('sqlite:///luxury_items.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Initialize scrapers
scrapers = {
    "ebay": EbayScraper()
    # Add other scrapers here
}

# Initialize email service
email_service = EmailService()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/search")
async def search(
    request: Request,
    search_term: str = Form(...),
    category: str = Form(...),
    threshold: float = Form(0.2)
):
    all_items = []
    tasks = []

    # Create tasks for each scraper
    for scraper in scrapers.values():
        tasks.append(scraper.scrape(search_term))

    # Run all scrapers concurrently
    results = await asyncio.gather(*tasks)
    
    # Combine results
    for items in results:
        all_items.extend(items)

    # Get market values and identify deals
    for item in all_items:
        if item.get('model'):
            market_value = await scrapers['ebay'].get_market_value(item['model'])
            if market_value:
                item['market_value'] = market_value
                item['is_good_deal'] = scrapers['ebay'].is_good_deal(
                    item['price'],
                    market_value,
                    threshold
                )
                
                # Send email for good deals
                if item['is_good_deal']:
                    await email_service.send_deal_alert(item)

    # Store results in database
    db = next(get_db())
    for item in all_items:
        db_item = Item(
            title=item['title'],
            price=item['price'],
            url=item['url'],
            platform=item['platform'],
            category=item['category'],
            image_url=item['image_url'],
            condition=item['condition'],
            seller=item['seller'],
            description=item['description'],
            model=item['model'],
            brand=item['brand'],
            is_good_deal=item.get('is_good_deal', False),
            market_value=item.get('market_value'),
            deal_percentage=((item.get('market_value', 0) - item['price']) / item.get('market_value', 1)) if item.get('market_value') else None
        )
        db.add(db_item)
    
    db.commit()

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "items": all_items,
            "search_term": search_term,
            "category": category
        }
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 