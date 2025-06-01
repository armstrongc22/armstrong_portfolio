import streamlit as st
import yagmail
from typing import Dict

class EmailService:
    def __init__(self):
        # Get credentials from Streamlit secrets
        self.email = st.secrets["email"]["address"]
        self.password = st.secrets["email"]["password"]
        self.yag = yagmail.SMTP(self.email, self.password)

    def format_deal_email(self, item: Dict) -> tuple:
        """Format the email content for a deal alert"""
        subject = f"🔥 Great Deal Alert: {item['brand']} {item['model']}"
        
        deal_percentage = ((item['market_value'] - item['price']) / item['market_value']) * 100
        
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; background-color: white; border: 2px solid gold;">
            <h1 style="color: gold; text-align: center;">Luxury Deal Alert! 🌟</h1>
            <div style="text-align: center;">
                <img src="{item['image_url']}" style="max-width: 300px; margin: 20px 0;">
            </div>
            <h2 style="color: #333;">{item['title']}</h2>
            <p style="color: #666; font-size: 18px;">
                <strong>Price:</strong> ${item['price']:,.2f}<br>
                <strong>Market Value:</strong> ${item['market_value']:,.2f}<br>
                <strong>Savings:</strong> {deal_percentage:.1f}%<br>
                <strong>Condition:</strong> {item['condition']}<br>
                <strong>Seller:</strong> {item['seller']}<br>
                <strong>Platform:</strong> {item['platform']}
            </p>
            <div style="text-align: center; margin-top: 20px;">
                <a href="{item['url']}" style="background-color: gold; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">
                    View Deal
                </a>
            </div>
            <p style="color: #999; font-size: 12px; margin-top: 20px; text-align: center;">
                This is an automated alert from your Luxury Item Price Tracker
            </p>
        </div>
        """
        
        return subject, html_content

    async def send_deal_alert(self, item: Dict) -> bool:
        """Send an email alert for a good deal"""
        try:
            subject, content = self.format_deal_email(item)
            self.yag.send(
                to=self.email,
                subject=subject,
                contents=content
            )
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False 