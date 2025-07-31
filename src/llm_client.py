import google.generativeai as genai
import json
import os
import re
from typing import Dict, Any

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    async def parse_task(self, question_text: str) -> Dict[str, Any]:
        """Parse the input task into structured format"""
        prompt = f"""
        Analyze this data analysis request and break it down into structured tasks:
        
        {question_text}
        
        Return a JSON object with this structure:
        {{
            "data_source": "URL or description of data source",
            "tasks": [
                {{
                    "type": "numerical|text|correlation|visualization",
                    "question": "the specific question",
                    "details": "additional details for processing"
                }}
            ]
        }}
        
        Task types:
        - numerical: Questions asking for counts, numbers, calculations
        - text: Questions asking for names, titles, or text responses
        - correlation: Questions about relationships between variables
        - visualization: Requests for charts, plots, graphs
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback parsing
                return self._fallback_parse(question_text)
        except Exception as e:
            return self._fallback_parse(question_text)
    
    def _fallback_parse(self, question_text: str) -> Dict[str, Any]:
        """Fallback parsing if LLM fails"""
        # Simple regex-based parsing
        tasks = []
        lines = question_text.split('\n')
        
        for line in lines:
            if re.match(r'^\d+\.', line.strip()):
                if 'correlation' in line.lower():
                    task_type = 'correlation'
                elif 'draw' in line.lower() or 'plot' in line.lower() or 'chart' in line.lower():
                    task_type = 'visualization'
                elif 'how many' in line.lower() or 'count' in line.lower():
                    task_type = 'numerical'
                else:
                    task_type = 'text'
                
                tasks.append({
                    'type': task_type,
                    'question': line.strip(),
                    'details': ''
                })
        
        url = self._extract_url_from_text(question_text)
        return {
            'data_source': url or 'unknown',
            'tasks': tasks
        }
    
    def _extract_url_from_text(self, text: str) -> str:
        """Extract URL from question text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
        matches = re.findall(url_pattern, text)
        return matches[0] if matches else None