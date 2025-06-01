from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime

class BaseScraper(ABC):
    def __init__(self, category: str = "watches"):
        self.category = category
        self.platform_name = self.__class__.__name__

    @abstractmethod
    async def scrape(self, search_term: str) -> List[Dict]:
        """
        Scrape items from the platform
        
        Args:
            search_term: The item to search for
            
        Returns:
            List of dictionaries containing item details
        """
        pass

    @abstractmethod
    async def get_market_value(self, model: str) -> Optional[float]:
        """
        Get the current market value for a specific model
        
        Args:
            model: The model name/number to look up
            
        Returns:
            Average market value if found, None otherwise
        """
        pass

    def is_good_deal(self, price: float, market_value: float, threshold: float = 0.2) -> bool:
        """
        Determine if a price represents a good deal
        
        Args:
            price: Listed price
            market_value: Current market value
            threshold: Minimum percentage below market value to be considered a good deal
            
        Returns:
            Boolean indicating if it's a good deal
        """
        if not market_value:
            return False
        return (market_value - price) / market_value >= threshold

    def format_item_data(self, raw_data: Dict) -> Dict:
        """
        Standardize the item data format
        """
        return {
            "title": raw_data.get("title", ""),
            "price": raw_data.get("price", 0.0),
            "url": raw_data.get("url", ""),
            "platform": self.platform_name,
            "category": self.category,
            "timestamp": datetime.now().isoformat(),
            "image_url": raw_data.get("image_url", ""),
            "condition": raw_data.get("condition", ""),
            "seller": raw_data.get("seller", ""),
            "description": raw_data.get("description", ""),
            "model": raw_data.get("model", ""),
            "brand": raw_data.get("brand", "")
        } 