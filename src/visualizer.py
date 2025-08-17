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
from src.llm_client import GeminiClient


class ChartGenerator:
    def __init__(self):
        # Set seaborn as primary style - modern and clean
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)
        sns.set_palette("husl")
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
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
        """Generate Python code for chart creation using LLM with seaborn focus"""
        prompt = f"""
        You are a data visualization code generator specializing in seaborn. Generate Python code to create a chart based on the task and data.

        SAMPLE DATA STRUCTURE:
        - Shape: {sample_data['shape']}
        - Columns: {sample_data['columns']}
        - Data types: {sample_data['dtypes']}
        - Column samples: {json.dumps(sample_data['column_samples'], indent=2)}

        TASK: {task['question']}
        
        REQUIREMENTS:
        1. The DataFrame is already loaded as 'df'
        2. PRIMARY: Use seaborn functions (sns.scatterplot, sns.barplot, sns.lineplot, sns.boxplot, sns.heatmap, etc.)
        3. SECONDARY: Use matplotlib.pyplot only for figure management and saving
        4. Handle missing/invalid data gracefully with df.dropna() or fillna()
        5. Use figsize=(12, 8) for good proportions
        6. Add appropriate titles and labels using plt.title(), plt.xlabel(), plt.ylabel()
        7. Use seaborn's built-in styling and color palettes
        8. Save the plot using plt.savefig() to SAVE_PATH
        9. Clear the plot with plt.close() after saving
        10. Set DPI to 100 for good quality

        SEABORN CHART TYPES TO PREFER:
        - Scatter plots: sns.scatterplot(data=df, x='col1', y='col2', hue='category')
        - Bar charts: sns.barplot(data=df, x='category', y='value')
        - Line plots: sns.lineplot(data=df, x='time', y='value', hue='group')
        - Box plots: sns.boxplot(data=df, x='category', y='value')
        - Violin plots: sns.violinplot(data=df, x='category', y='value')
        - Heatmaps: sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        - Histograms: sns.histplot(data=df, x='value', kde=True)
        - Count plots: sns.countplot(data=df, x='category')
        - Regression plots: sns.regplot(data=df, x='x', y='y') or sns.lmplot()

        EXAMPLE CODE STRUCTURE:
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # Data cleaning and preparation
        df_clean = df.dropna()  # or appropriate cleaning
        
        # Create the visualization using seaborn
        plt.figure(figsize=(12, 8))
        
        # Use seaborn function (example)
        sns.scatterplot(data=df_clean, x='column1', y='column2', hue='category', s=60, alpha=0.7)
        
        # Add labels and formatting
        plt.title('Your Chart Title', fontsize=16, fontweight='bold')
        plt.xlabel('X Label', fontsize=12)
        plt.ylabel('Y Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(SAVE_PATH, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close()
        ```

        Generate ONLY the Python code focusing on seaborn functions, no explanations:
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
            
            # Convert Windows path to forward slashes
            data_path_unix = data_path.replace('\\', '/')
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_file:
                    img_path = img_file.name
                    img_path_unix = img_path.replace('\\', '/')
                
                # Write the chart generation code with proper path handling
                full_code = f"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("husl")

# Load the data (using forward slashes for Windows compatibility)
df = pd.read_csv('{data_path_unix}')

# Set the save path
SAVE_PATH = '{img_path_unix}'

# Generated chart code
{code}

print("Chart saved successfully")
"""
                code_file.write(full_code)
                code_file_path = code_file.name
            
            print(f"Chart code saved to: {code_file_path}")
            print(f"Data path: {data_path_unix}")
            print(f"Image will be saved to: {img_path_unix}")
            
            # Execute the code
            result = subprocess.run(
                [sys.executable, code_file_path],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout for chart generation
            )
            
            # Clean up code file and data file
            try:
                os.unlink(code_file_path)
                os.unlink(data_path)
            except OSError:
                pass
            
            if result.returncode != 0:
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                raise Exception(f"Chart generation failed: {result.stderr}")
            
            # Read the generated image and convert to base64
            if os.path.exists(img_path):
                with open(img_path, 'rb') as img_file:
                    img_data = img_file.read()
                    base64_image = base64.b64encode(img_data).decode()
                
                # Clean up image file
                try:
                    os.unlink(img_path)
                except OSError:
                    pass
                
                # Check size and compress if needed
                if len(base64_image) > 100000:
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

    # Convenience methods for common seaborn charts
    async def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None) -> str:
        """Create a scatter plot using seaborn"""
        task = {
            'question': f'Create a scatter plot of {x_col} vs {y_col}' + (f' colored by {hue_col}' if hue_col else ''),
            'type': 'visualization'
        }
        return await self.generate_chart_with_llm(data, task)
    
    async def create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Create a bar chart using seaborn"""
        task = {
            'question': f'Create a bar chart showing {y_col} by {x_col}',
            'type': 'visualization'
        }
        return await self.generate_chart_with_llm(data, task)
    
    async def create_correlation_heatmap(self, data: pd.DataFrame) -> str:
        """Create a correlation heatmap using seaborn"""
        task = {
            'question': 'Create a correlation heatmap of all numeric columns',
            'type': 'visualization'
        }
        return await self.generate_chart_with_llm(data, task)
    
    async def create_box_plot(self, data: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Create a box plot using seaborn"""
        task = {
            'question': f'Create a box plot showing distribution of {y_col} by {x_col}',
            'type': 'visualization'
        }
        return await self.generate_chart_with_llm(data, task)
