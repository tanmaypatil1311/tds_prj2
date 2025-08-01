# ## 7. Visualization with Matplotlib

# ### src/visualizer.py
# ```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
import io
import subprocess
import tempfile
import json
import os
import sys
from typing import Dict
from llm_client import GeminiClient

class ChartGenerator:
    def __init__(self):
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.llm = GeminiClient()
    
    async def generate_chart_with_llm(self, data: pd.DataFrame, task: Dict) -> str:
        """Generate chart using LLM-generated code and return as base64 data URI"""
        try:
            # Create sample data for LLM
            sample_data = self._create_sample_data(data)
            
            # Generate visualization code using LLM
            chart_code = await self._generate_chart_code(sample_data, task)
            
            # Execute the generated code and get base64 image
            base64_image = await self._execute_chart_code(chart_code, data)
            
            return base64_image
            
        except Exception as e:
            raise Exception(f"LLM-based chart generation failed: {str(e)}")
    
    def _create_sample_data(self, data: pd.DataFrame) -> Dict:
        """Create sample data representation for LLM"""
        sample = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'sample_rows': data.head(5).to_dict('records'),
            'column_samples': {}
        }
        
        # Add column statistics for numeric columns
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64'] or pd.to_numeric(data[col], errors='coerce').notna().sum() > 0:
                numeric_data = pd.to_numeric(data[col], errors='coerce')
                if numeric_data.notna().sum() > 0:
                    sample['column_samples'][col] = {
                        'type': 'numeric',
                        'min': float(numeric_data.min()),
                        'max': float(numeric_data.max()),
                        'mean': float(numeric_data.mean())
                    }
            else:
                unique_vals = data[col].unique()[:10]  # Limit to 10 unique values
                sample['column_samples'][col] = {
                    'type': 'categorical',
                    'unique_values': list(unique_vals)
                }
        
        return sample
    
    async def _generate_chart_code(self, sample_data: Dict, task: Dict) -> str:
        """Generate Python code for chart creation using LLM"""
        prompt = f"""
        You are a data visualization code generator. Generate Python code to create a chart based on the task and data.

        SAMPLE DATA STRUCTURE:
        - Shape: {sample_data['shape']}
        - Columns: {sample_data['columns']}
        - Data types: {sample_data['dtypes']}
        - Column samples: {json.dumps(sample_data['column_samples'], indent=2)}

        TASK: {task['question']}
        
        REQUIREMENTS:
        1. The DataFrame is already loaded as 'df'
        2. Import matplotlib.pyplot as plt, seaborn as sns, numpy as np, pandas as pd
        3. Create the requested visualization (scatter plot, bar chart, line plot, etc.)
        4. Handle missing/invalid data gracefully
        5. Add appropriate labels, title, and formatting
        6. Use figsize=(10, 6) for good proportions
        7. If regression line requested, add it with specified color/style
        8. Save the plot using plt.savefig() to a specified path
        9. Clear the plot with plt.close() after saving
        10. Set DPI to 100 for good quality but reasonable file size

        EXAMPLE CODE STRUCTURE:
        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        
        # Data cleaning and preparation
        # ... your cleaning code here ...
        
        # Create the visualization
        plt.figure(figsize=(10, 6))
        
        # ... your plotting code here ...
        
        plt.title('Your Chart Title')
        plt.xlabel('X Label')
        plt.ylabel('Y Label')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig(SAVE_PATH, dpi=100, bbox_inches='tight')
        plt.close()
        ```

        Generate ONLY the Python code, no explanations:
        """
        
        try:
            response = await self.llm.generate_content(prompt)
            
            # Extract code from response
            code = self._extract_code_from_response(response.text)
            return code
            
        except Exception as e:
            raise Exception(f"Chart code generation failed: {str(e)}")
    
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
            if (stripped.startswith(('import ', 'from ', 'plt.', 'sns.', 'df', '#')) or 
                '=' in stripped or 
                stripped.startswith(('if ', 'for ', 'while ', 'def ', 'try:', 'except'))):
                code_lines.append(line)
        
        return '\n'.join(code_lines)
    
    async def _execute_chart_code(self, code: str, data: pd.DataFrame) -> str:
        """Execute the generated chart code and return base64 encoded image"""
        try:
            # Create temporary files
            data_path = self._save_temp_data(data)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
                    img_path = img_file.name
                
                # Write the chart generation code
                full_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('{data_path}')

# Set the save path
SAVE_PATH = '{img_path}'

# Generated chart code
{code}

print("Chart saved successfully")
"""
                code_file.write(full_code)
                code_file_path = code_file.name
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, code_file_path],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for chart generation
            )
            
            # Clean up code file
            os.unlink(code_file_path)
            os.unlink(data_path)
            
            if result.returncode != 0:
                raise Exception(f"Chart generation failed: {result.stderr}")
            
            # Read the generated image and convert to base64
            if os.path.exists(img_path):
                with open(img_path, 'rb') as img_file:
                    img_data = img_file.read()
                    base64_image = base64.b64encode(img_data).decode()
                
                # Clean up image file
                os.unlink(img_path)
                
                # Check size (under 100KB)
                if len(base64_image) > 100000:
                    # Try to compress or reduce quality
                    base64_image = await self._compress_image(base64_image)
                
                return f"data:image/png;base64,{base64_image}"
            else:
                raise Exception("Chart image was not generated")
                
        except subprocess.TimeoutExpired:
            raise Exception("Chart generation timeout")
        except Exception as e:
            raise Exception(f"Chart execution error: {str(e)}")
    
    def _save_temp_data(self, data: pd.DataFrame) -> str:
        """Save DataFrame to temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            data.to_csv(temp_file.name, index=False)
            return temp_file.name
    
    async def _compress_image(self, base64_image: str) -> str:
        """Compress image if it's too large"""
        try:
            import base64
            from PIL import Image
            import io
            
            # Decode base64 to image
            img_data = base64.b64decode(base64_image)
            img = Image.open(io.BytesIO(img_data))
            
            # Reduce quality/size
            output = io.BytesIO()
            img.save(output, format='PNG', optimize=True, quality=70)
            compressed_data = output.getvalue()
            
            return base64.b64encode(compressed_data).decode()
        
        except Exception:
            # If compression fails, truncate the original
            return base64_image[:100000] if len(base64_image) > 100000 else base64_image