import pandas as pd
import numpy as np
import subprocess
import tempfile
import json
import os
from typing import Any, Dict, List
from .llm_client import GeminiClient

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.llm = GeminiClient()
    
    async def structure_data(self, html_content: str, task_info: Dict) -> pd.DataFrame:
        """Convert scraped HTML to structured data - works with any website"""
        from .scraper import WebScraper
        from bs4 import BeautifulSoup
        scraper = WebScraper()
        
        # Extract tables from HTML
        tables = scraper.extract_tables(html_content)
        
        if not tables:
            # If no tables, try to extract other structured data
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try to find lists, divs with data, etc.
            structured_content = self._extract_structured_content(soup)
            if structured_content:
                return pd.DataFrame(structured_content)
            else:
                raise Exception("No structured data found in scraped content")
        
        # Find the most relevant table based on size and content
        main_table = self._select_best_table(tables, task_info)
        
        # Clean column names
        main_table.columns = main_table.columns.astype(str)
        
        self.data = main_table
        return main_table
    
    async def analyze_with_llm_code(self, data: pd.DataFrame, task: Dict) -> Any:
        """Generate and execute code using LLM for specific analysis task"""
        try:
            # Create sample data (first 10 rows + column info)
            sample_data = self._create_sample_data(data)
            
            # Generate analysis code using LLM
            analysis_code = await self._generate_analysis_code(sample_data, task)
            
            # Execute the generated code
            result = await self._execute_analysis_code(analysis_code, data)
            
            return result
            
        except Exception as e:
            raise Exception(f"LLM-based analysis failed: {str(e)}")
    
    def _create_sample_data(self, data: pd.DataFrame) -> Dict:
        """Create sample data representation for LLM"""
        sample = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'sample_rows': data.head(10).to_dict('records'),
            'column_samples': {}
        }
        
        # Add unique values for categorical columns (limited)
        for col in data.columns:
            if data[col].dtype == 'object':
                unique_vals = data[col].unique()[:20]  # Limit to 20 unique values
                sample['column_samples'][col] = list(unique_vals)
        
        return sample
    
    async def _generate_analysis_code(self, sample_data: Dict, task: Dict) -> str:
        """Generate Python code using LLM for the specific analysis task"""
        prompt = f"""
        You are a data analysis code generator. Given the sample data and task, generate Python code that performs the analysis.

        SAMPLE DATA STRUCTURE:
        - Shape: {sample_data['shape']}
        - Columns: {sample_data['columns']}
        - Data types: {sample_data['dtypes']}
        - Sample rows: {json.dumps(sample_data['sample_rows'][:3], indent=2)}

        TASK: {task['question']}
        TASK TYPE: {task['type']}

        REQUIREMENTS:
        1. The DataFrame is already loaded as 'df'
        2. Import necessary libraries at the top (pandas, numpy, etc.)
        3. Return the final result as 'result' variable
        4. Handle missing/invalid data gracefully
        5. For numerical results, return int/float
        6. For text results, return string
        7. For correlations, return float rounded to 6 decimals
        8. Clean and convert data types as needed

        EXAMPLE CODE STRUCTURE:
        ```python
        import pandas as pd
        import numpy as np
        from datetime import datetime

        # Data cleaning and preparation
        # ... your cleaning code here ...

        # Analysis logic
        # ... your analysis code here ...

        # Set the final result
        result = your_calculated_result
        ```

        Generate ONLY the Python code, no explanations:
        """
        
        try:
            response = await self.llm.generate_content(prompt)
            
            # Extract code from response
            code = self._extract_code_from_response(response.text)
            return code
            
        except Exception as e:
            raise Exception(f"Code generation failed: {str(e)}")
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from LLM response"""
        import re
        
        # Try to find code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response_text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Try to find code without markdown
        code_blocks = re.findall(r'```\n(.*?)\n```', response_text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # If no code blocks, assume the entire response is code
        lines = response_text.strip().split('\n')
        # Remove any explanation lines that don't look like code
        code_lines = []
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith(('import ', 'from ', 'df', 'result', '#')) or 
                '=' in stripped or 
                stripped.startswith(('if ', 'for ', 'while ', 'def ', 'try:', 'except'))):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    async def _execute_analysis_code(self, code: str, data: pd.DataFrame) -> Any:
        """Execute the generated analysis code safely"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                # Write the analysis code
                full_code = f"""
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('{self._save_temp_data(data)}')

# Generated analysis code
{code}

# Output result as JSON
print(json.dumps({{'result': result, 'type': str(type(result).__name__)}}))
"""
                code_file.write(full_code)
                code_file_path = code_file.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, code_file_path],
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            # Clean up
            os.unlink(code_file_path)
            
            if result.returncode != 0:
                raise Exception(f"Code execution failed: {result.stderr}")
            
            # Parse result
            try:
                output = json.loads(result.stdout.strip())
                return output['result']
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                return result.stdout.strip()
                
        except subprocess.TimeoutExpired:
            raise Exception("Analysis code execution timeout")
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")
    
    def _save_temp_data(self, data: pd.DataFrame) -> str:
        """Save DataFrame to temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            data.to_csv(temp_file.name, index=False)
            return temp_file.name
    
    # Keep only the utility methods for data structure handling
    def _select_best_table(self, tables: List[pd.DataFrame], task_info: Dict) -> pd.DataFrame:
        """Select the most relevant table from multiple tables"""
        if len(tables) == 1:
            return tables[0]
        
        # Score tables based on size and relevance
        best_table = None
        best_score = 0
        
        for table in tables:
            score = 0
            
            # Size factor (larger tables often contain main data)
            score += len(table) * 0.1
            score += len(table.columns) * 0.05
            
            # Content relevance (check if columns match expected data types)
            for col in table.columns:
                col_str = str(col).lower()
                # Look for common data indicators
                if any(keyword in col_str for keyword in ['rank', 'name', 'title', 'year', 'value', 'amount', 'price', 'score']):
                    score += 10
                if any(keyword in col_str for keyword in ['gross', 'revenue', 'sales', 'budget', 'profit']):
                    score += 15
            
            if score > best_score:
                best_score = score
                best_table = table
        
        return best_table or tables[0]  # Fallback to first table
    
    def _extract_structured_content(self, soup) -> List[Dict]:
        """Extract structured content from non-table elements"""
        structured_data = []
        
        # Try to find structured lists
        lists = soup.find_all(['ul', 'ol'])
        for lst in lists:
            items = lst.find_all('li')
            if len(items) > 5:  # Only consider substantial lists
                for item in items:
                    # Extract text and try to parse structured info
                    text = item.get_text(strip=True)
                    if text:
                        # Try to extract key-value pairs or structured info
                        parsed_item = self._parse_list_item(text)
                        if parsed_item:
                            structured_data.append(parsed_item)
        
        return structured_data
    
    def _parse_list_item(self, text: str) -> Dict:
        """Parse individual list items for structured data"""
        import re
        # This is a basic parser - can be enhanced based on common patterns
        item = {'text': text}
        
        # Extract numbers
        numbers = re.findall(r'\d+(?:,\d+)*(?:\.\d+)?', text)
        if numbers:
            item['numbers'] = numbers
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            item['years'] = years
        
        # Extract currency amounts
        currency = re.findall(r'\$[\d,]+(?:\.\d+)?[BMK]?', text)
        if currency:
            item['currency'] = currency
        
        return item