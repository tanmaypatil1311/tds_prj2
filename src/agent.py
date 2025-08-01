import asyncio
import json
import re
from typing import List, Dict, Any
from llm_client import GeminiClient
from scraper import WebScraper
from analyzer import DataAnalyzer
from visualizer import ChartGenerator

class DataAnalystAgent:
    def __init__(self):
        self.llm = GeminiClient()
        self.scraper = WebScraper()
        self.analyzer = DataAnalyzer()
        self.visualizer = ChartGenerator()
    
    async def process_request(self, question_text: str) -> List[Any]:
        """Main processing pipeline"""
        try:
            # Step 1: Parse the task using LLM
            task_breakdown = await self.llm.parse_task(question_text)
            print(f"Parsed task breakdown: {json.dumps(task_breakdown, indent=2)}")
            # Step 2: Extract data source URL and scrape
            url = self.extract_url(question_text)
            if url:
                raw_data = await self.scraper.scrape_url(url)
                print(f"Raw data scraped from {url}")
                structured_data = await self.analyzer.structure_data(raw_data, task_breakdown)
            else:
                structured_data = None
            
            print(f"Structured data: {structured_data}")
            # Step 3: Process each question/task using LLM-generated code
            results = []
            for task in task_breakdown.get('tasks', []):
                if task['type'] == 'visualization':
                    # Handle visualization separately (needs image generation)
                    print(f"Generating chart for task: {task['question']}")
                    result = await self.visualizer.generate_chart_with_llm(structured_data, task)
                else:
                    # Use LLM to generate and execute analysis code
                    print(f"Analyzing task: {task['question']}")
                    result = await self.analyzer.analyze_with_llm_code(structured_data, task)
                
                results.append(result)
            
            return results
        
        except Exception as e:
            raise Exception(f"Processing failed: {str(e)}")
    
    def extract_url(self, text: str) -> str:
        """Extract URL from question text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        matches = re.findall(url_pattern, text)
        print(f"Extracting URL from text: {matches}")
        return matches[0] if matches else None