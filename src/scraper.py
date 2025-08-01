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
        print(f"Scraping URL: {url}")
        async with async_playwright() as p:
            print("Launching browser...")
            # Launch browser (headless for production)
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            print("Browser launched, navigating to URL...") 
            try:
                # Navigate to URL
                await page.goto(url, wait_until='networkidle')
                print("Page loaded, waiting for content...")
                # Get page content
                content = await page.content()
                
                # Close browser
                await browser.close()
                print("Browser closed, returning content...")
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
                # Use pd.read_html directly on the table string
                df_list = pd.read_html(str(table))
                if df_list:  # Check if list is not empty
                    for df in df_list:
                        if not df.empty:  # Only add non-empty DataFrames
                            dataframes.append(df)
            except Exception as e:
                print(f"Failed to parse table: {e}")
                continue
        
        print(f"Extracted {len(dataframes)} tables from HTML content.")
        return dataframes

