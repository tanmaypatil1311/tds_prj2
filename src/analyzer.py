import sys
import pandas as pd
import numpy as np
import subprocess
import tempfile
import json
import os
from typing import Any, Dict, List

from scraper import WebScraper
from llm_client import GeminiClient  # Commented out for testing

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.llm = GeminiClient()  # Commented out for testing
    
    async def structure_data(self, html_content: str, task_info: Dict) -> pd.DataFrame:
        """Convert scraped HTML to structured data - works with any website"""
        print("Structuring data from HTML content...")
        # from scraper import WebScraper
        from bs4 import BeautifulSoup
        scraper = WebScraper()
        
        # Extract tables from HTML
        tables = scraper.extract_tables(html_content)
        
        # Fix: Properly check for empty tables list or all empty DataFrames
        if len(tables) == 0:
            print("No tables found, trying to extract other structured data...")
            # If no tables, try to extract other structured data
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Try to find lists, divs with data, etc.
            structured_content = self._extract_structured_content(soup)
            if structured_content:
                return pd.DataFrame(structured_content)
            else:
                raise Exception("No structured data found in scraped content")
        
        # Filter out empty tables
        non_empty_tables = []
        for table in tables:
            if not table.empty and len(table.columns) > 0:
                non_empty_tables.append(table)
        
        if len(non_empty_tables) == 0:
            raise Exception("All extracted tables are empty")
        
        # Find the most relevant table based on size and content
        main_table = self._select_best_table(non_empty_tables, task_info)
        
        print(f"Selected main table with shape: {main_table.shape}")    
        
        self.data = main_table
        print(f"Structured data shape: {main_table.shape}")
        print(f"Columns: {main_table.columns.tolist()}")
        return main_table
    
    async def analyze_with_llm_code(self, data: pd.DataFrame, task: Dict) -> Any:
        """Generate and execute code using LLM for specific analysis task"""
        try:
            print(f"Analyzing task: {task['question']}")
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
        print("Creating sample data for LLM...")
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
        print("Generating analysis code using LLM...")
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
            # For testing without LLM client
            response = await self.llm.generate_content(prompt)
            code = self._extract_code_from_response(response.text)
            
            # Placeholder code for testing
#             code = """
# import pandas as pd
# import numpy as np

# # Simple test analysis
# result = len(df)
# """
            return code
            
        except Exception as e:
            raise Exception(f"Code generation failed: {str(e)}")
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from LLM response"""
        print("Extracting code from LLM response...")
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
            print("Executing generated analysis code...")
            
            # Save data to temp file and get the path
            temp_data_path = self._save_temp_data(data)
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                # Write the analysis code with proper path escaping
                full_code = f"""
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data (using raw string to handle Windows paths)
df = pd.read_csv(r'{temp_data_path}')

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
            
            # Clean up temporary files
            try:
                os.unlink(code_file_path)
                os.unlink(temp_data_path)
            except OSError:
                pass  # Ignore cleanup errors
            
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
        print("Saving DataFrame to temporary CSV file...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            data.to_csv(temp_file.name, index=False)
            return temp_file.name
    
    def _select_best_table(self, tables: List[pd.DataFrame], task_info: Dict) -> pd.DataFrame:
        """Select the most relevant table from multiple tables"""
        print("Selecting the best table from extracted tables...")
        
        if len(tables) == 1:
            return tables[0]
        
        # Score tables based on size and relevance
        best_table = None
        best_score = 0
        
        for table in tables:
            # Skip empty tables (should already be filtered, but double-check)
            if table.empty:
                continue
                
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
        
        print(f"Best table selected with score: {best_score}")
        
        # Return best table or first table as fallback
        if best_table is not None:
            return best_table
        else:
            return tables[0]
    
    def _extract_structured_content(self, soup) -> List[Dict]:
        """Extract structured content from non-table elements"""
        print("Extracting structured content from HTML...")
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
        print(f"Parsing list item: {text}")
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
