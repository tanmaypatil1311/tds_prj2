##analyzer.py
import sys
import pandas as pd
import numpy as np
import subprocess
import tempfile
import json
import os
from typing import Any, Dict, List


from src.llm_client import GeminiClient
from src.scraper import WebScraper

class DataAnalyzer:
    def __init__(self):
        self.data = None
        self.llm = GeminiClient()
    
    async def structure_data(self, html_content: str, task_info: Dict) -> pd.DataFrame:
        """Convert scraped HTML to structured data - works with any website"""
        print("Structuring data from HTML content...")
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
        
        # Find the most relevant table based on tasks using LLM
        main_table = await self._select_best_table_with_llm(non_empty_tables, task_info)
        
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
            response = await self.llm.generate_content(prompt)
            code = self._extract_code_from_response(response.text)
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
            print(f"Generated code saved to: {code_file_path}")
            
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
    
    async def _select_best_table_with_llm(self, tables: List[pd.DataFrame], task_info: Dict) -> pd.DataFrame:
        """Select the most relevant table using LLM analysis"""
        print("Using LLM to select the best table for the tasks...")
        
        if len(tables) == 1:
            print("Only one table found, returning it.")
            return tables[0]
        
        try:
            # Prepare table metadata for LLM
            table_metadata = self._prepare_table_metadata(tables)
            
            # Get LLM's decision
            selected_table_index = await self._get_llm_table_selection(table_metadata, task_info)
            
            # Validate the selection
            if 0 <= selected_table_index < len(tables):
                print(f"LLM selected table {selected_table_index}")
                return tables[selected_table_index]
            else:
                print(f"Invalid LLM selection {selected_table_index}, falling back to largest table")
                return self._fallback_table_selection(tables)
                
        except Exception as e:
            print(f"LLM table selection failed: {str(e)}, falling back to heuristic selection")
            return self._fallback_table_selection(tables)
    
    def _prepare_table_metadata(self, tables: List[pd.DataFrame]) -> List[Dict]:
        """Prepare lightweight metadata for each table"""
        print("Preparing table metadata for LLM analysis...")
        metadata = []
        
        for i, table in enumerate(tables):
            # Get basic table info
            table_info = {
                'table_number': i,
                'shape': table.shape,
                'columns': list(table.columns),
                'column_count': len(table.columns),
                'row_count': len(table),
                'data_types': {}
            }
            
            # Add data type information
            for col in table.columns:
                dtype_str = str(table[col].dtype)
                # Simplify dtype for LLM
                if 'int' in dtype_str:
                    table_info['data_types'][col] = 'integer'
                elif 'float' in dtype_str:
                    table_info['data_types'][col] = 'numeric'
                elif 'object' in dtype_str:
                    table_info['data_types'][col] = 'text'
                elif 'datetime' in dtype_str:
                    table_info['data_types'][col] = 'date'
                else:
                    table_info['data_types'][col] = 'other'
            
            # Add sample values for context (first few non-null values per column)
            table_info['sample_values'] = {}
            for col in table.columns:
                non_null_values = table[col].dropna()
                if len(non_null_values) > 0:
                    # Get first 3 unique values
                    sample_vals = non_null_values.unique()[:3]
                    table_info['sample_values'][col] = [str(val) for val in sample_vals]
                else:
                    table_info['sample_values'][col] = []
            
            metadata.append(table_info)
        
        return metadata
    
    async def _get_llm_table_selection(self, table_metadata: List[Dict], task_info: Dict) -> int:
        """Use LLM to select the best table based on tasks"""
        print("Asking LLM to select the best table...")
        
        # Extract tasks from task_info
        tasks = task_info.get('tasks', [])
        if not tasks:
            # If no tasks in task_info, check if it's a single task
            if 'question' in task_info:
                tasks = [task_info]
        
        # Prepare the prompt
        prompt = f"""
        You are a data analyst tasked with selecting the most appropriate table for analysis.

        AVAILABLE TABLES:
        {json.dumps(table_metadata, indent=2)}

        ANALYSIS TASKS TO PERFORM:
        {json.dumps(tasks, indent=2)}

        INSTRUCTIONS:
        1. Analyze which table contains the data most relevant to ALL the given tasks
        2. Consider column names, data types, sample values, and table size
        3. Look for tables that contain the key data needed for the analysis tasks
        4. Prioritize tables with more relevant columns and adequate data volume
        5. Return ONLY the table number (integer) that best matches the requirements

        IMPORTANT: 
        - Return only a single integer (0, 1, 2, etc.) representing the table number
        - Do not include any explanation or additional text
        - Choose the table that can best support all or most of the analysis tasks

        Best table number:
        """
        
        try:
            response = await self.llm.generate_content(prompt)
            
            # Extract table number from response
            table_number = self._extract_table_number(response.text)
            print(f"LLM selected table number: {table_number}")
            return table_number
            
        except Exception as e:
            raise Exception(f"LLM table selection failed: {str(e)}")
    
    def _extract_table_number(self, response_text: str) -> int:
        """Extract table number from LLM response"""
        import re
        
        # Clean the response
        response_text = response_text.strip()
        
        # Try to find a number at the start or end of response
        numbers = re.findall(r'\b\d+\b', response_text)
        
        if numbers:
            # Return the first number found
            return int(numbers[0])
        else:
            # If no number found, raise exception
            raise ValueError("No valid table number found in LLM response")
    
    def _fallback_table_selection(self, tables: List[pd.DataFrame]) -> pd.DataFrame:
        """Fallback table selection using heuristics when LLM fails"""
        print("Using fallback heuristic table selection...")
        
        # Simple heuristic: choose the table with most columns and rows
        best_table = None
        best_score = 0
        
        for table in tables:
            # Score based on size and column count
            score = len(table) * 0.1 + len(table.columns) * 10
            
            # Bonus for having numeric columns (often indicates data tables)
            numeric_cols = sum(1 for col in table.columns if table[col].dtype in ['int64', 'float64'])
            score += numeric_cols * 5
            
            if score > best_score:
                best_score = score
                best_table = table
        
        return best_table if best_table is not None else tables[0]
    
    def _select_best_table(self, tables: List[pd.DataFrame], task_info: Dict) -> pd.DataFrame:
        """Legacy method - kept for backward compatibility"""
        print("Using legacy table selection method...")
        return self._fallback_table_selection(tables)
    
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
