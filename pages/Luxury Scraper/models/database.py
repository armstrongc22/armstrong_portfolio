from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Item(Base):
    __tablename__ = 'items'

    id = Column(Integer, primary_key=True)
    title = Column(String)
    price = Column(Float)
    url = Column(String)
    platform = Column(String)
    category = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    image_url = Column(String)
    condition = Column(String)
    seller = Column(String)
    description = Column(String)
    model = Column(String)
    brand = Column(String)
    is_good_deal = Column(Boolean, default=False)
    market_value = Column(Float)
    deal_percentage = Column(Float)

class MarketValue(Base):
    __tablename__ = 'market_values'

    id = Column(Integer, primary_key=True)
    model = Column(String, unique=True)
    brand = Column(String)
    category = Column(String)
    current_value = Column(Float)
    last_updated = Column(DateTime, default=datetime.now)
    price_history = relationship("PriceHistory", back_populates="market_value")

class PriceHistory(Base):
    __tablename__ = 'price_history'

    id = Column(Integer, primary_key=True)
    market_value_id = Column(Integer, ForeignKey('market_values.id'))
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.now)
    market_value = relationship("MarketValue", back_populates="price_history")

# Create database engine
engine = create_engine('sqlite:///luxury_items.db')

# Create all tables
Base.metadata.create_all(engine) 