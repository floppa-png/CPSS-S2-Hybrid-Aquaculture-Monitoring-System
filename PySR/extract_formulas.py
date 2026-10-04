import json
import re
import codecs
import sys

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def extract():
    with open('c:/Users/vmj07/OneDrive/Desktop/PySR/PySR_Formulas_Aquaculture.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, cell in enumerate(data.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            print(f"=== Cell {i} ===")
            if "model_" in source:
                print(f"Found model definition in source.")
            
            outputs = cell.get('outputs', [])
            for out in outputs:
                if out.get('output_type') == 'stream':
                    text = "".join(out.get('text', []))
                    for line in text.split('\n'):
                        if "=>" in line or "Equation" in line or "Score" in line or "y =" in line or line.strip().startswith("1") or line.strip().startswith("2"):
                            print(line)
                elif out.get('output_type') == 'display_data' or out.get('output_type') == 'execute_result':
                    data_dict = out.get('data', {})
                    if 'text/plain' in data_dict:
                        text = "".join(data_dict['text/plain'])
                        for line in text.split('\n'):
                            if "=>" in line or "Equation" in line or "Score" in line or "y =" in line or line.strip().startswith("1") or line.strip().startswith("2"):
                                print(line)
                    if 'text/html' in data_dict:
                        text = "".join(data_dict['text/html'])
                        # Just a quick check if there's any dataframe table
                        if "Equation" in text:
                            print("Found HTML table with equations")

if __name__ == "__main__":
    extract()
