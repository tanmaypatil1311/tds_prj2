from typing import List
from playwright.async_api import async_playwright
import pandas as pd
from bs4 import BeautifulSoup
import asyncio

class WebScraper:
    def __init__(self):
        self.browser = None
        self.context = None
    
    async def scrape_url(self, url: str) -> str:
        """Scrape content from URL using Playwright"""
        async with async_playwright() as p:
            # Launch browser (headless for production)
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # Navigate to URL
                await page.goto(url, wait_until='networkidle')
                
                # Get page content
                content = await page.content()
                
                # Close browser
                await browser.close()
                
                return content
            
            except Exception as e:
                await browser.close()
                raise Exception(f"Scraping failed: {str(e)}")
    
    def extract_tables(self, html_content: str) -> List[pd.DataFrame]:
        """Extract tables from HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        dataframes = []
        for table in tables:
            try:
                df = pd.read_html(str(table))[0]
                dataframes.append(df)
            except:
                continue
        
        return dataframes