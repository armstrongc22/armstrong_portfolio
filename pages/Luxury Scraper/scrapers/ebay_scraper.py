import aiohttp
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import json
import re
from .base_scraper import BaseScraper

class EbayScraper(BaseScraper):
    def __init__(self, category: str = "watches"):
        super().__init__(category)
        self.base_url = "https://www.ebay.com"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

    async def scrape(self, search_term: str) -> List[Dict]:
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Format search URL
            search_url = f"{self.base_url}/sch/i.html?_nkw={search_term.replace(' ', '+')}&_sacat=0"
            
            try:
                async with session.get(search_url) as response:
                    if response.status != 200:
                        print(f"Error accessing eBay: Status {response.status}")
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    items = []

                    # Find all listing items
                    listings = soup.find_all('div', class_='s-item__info')
                    
                    for listing in listings:
                        try:
                            # Extract title
                            title_elem = listing.find('div', class_='s-item__title')
                            if not title_elem or 'Shop on eBay' in title_elem.text:
                                continue
                            
                            # Extract price
                            price_elem = listing.find('span', class_='s-item__price')
                            if not price_elem:
                                continue
                            
                            price_text = price_elem.text.replace('$', '').replace(',', '')
                            price = float(re.findall(r'\d+\.?\d*', price_text)[0])
                            
                            # Extract URL
                            url_elem = listing.find('a', class_='s-item__link')
                            url = url_elem['href'] if url_elem else None
                            
                            # Extract image
                            image_elem = listing.find('img', class_='s-item__image-img')
                            image_url = image_elem['src'] if image_elem else None
                            
                            # Extract condition
                            condition_elem = listing.find('span', class_='SECONDARY_INFO')
                            condition = condition_elem.text if condition_elem else 'Not specified'
                            
                            # Extract seller
                            seller_elem = listing.find('span', class_='s-item__seller-info-text')
                            seller = seller_elem.text if seller_elem else 'Unknown'
                            
                            # Try to extract brand and model from title
                            brand = None
                            model = None
                            title = title_elem.text
                            
                            # Common luxury watch brands
                            brands = ['Rolex', 'Patek Philippe', 'Audemars Piguet', 'Omega', 'Cartier', 'TAG Heuer', 'IWC']
                            for b in brands:
                                if b.lower() in title.lower():
                                    brand = b
                                    # Try to find model after brand name
                                    model_match = re.search(f"{b}\s+(.*?)(?:\s+|$)", title, re.IGNORECASE)
                                    if model_match:
                                        model = model_match.group(1)
                                    break
                            
                            item_data = {
                                "title": title,
                                "price": price,
                                "url": url,
                                "image_url": image_url,
                                "condition": condition,
                                "seller": seller,
                                "brand": brand,
                                "model": model,
                                "description": title  # Using title as description for eBay
                            }
                            
                            items.append(self.format_item_data(item_data))
                        except Exception as e:
                            print(f"Error processing listing: {str(e)}")
                            continue
                    
                    return items
            except Exception as e:
                print(f"Error during scraping: {str(e)}")
                return []

    async def get_market_value(self, model: str) -> Optional[float]:
        """
        Get market value by analyzing completed listings
        """
        async with aiohttp.ClientSession(headers=self.headers) as session:
            search_url = f"{self.base_url}/sch/i.html?_nkw={model.replace(' ', '+')}&_sacat=0&LH_Complete=1&LH_Sold=1"
            
            try:
                async with session.get(search_url) as response:
                    if response.status != 200:
                        return None
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    prices = []
                    for price_elem in soup.find_all('span', class_='s-item__price'):
                        try:
                            price_text = price_elem.text.replace('$', '').replace(',', '')
                            price = float(re.findall(r'\d+\.?\d*', price_text)[0])
                            prices.append(price)
                        except:
                            continue
                    
                    if not prices:
                        return None
                    
                    # Remove outliers (prices outside 1.5 IQR)
                    prices.sort()
                    q1 = prices[len(prices)//4]
                    q3 = prices[3*len(prices)//4]
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    filtered_prices = [p for p in prices if lower_bound <= p <= upper_bound]
                    
                    if not filtered_prices:
                        return None
                    
                    return sum(filtered_prices) / len(filtered_prices)
            except Exception as e:
                print(f"Error getting market value: {str(e)}")
                return None 