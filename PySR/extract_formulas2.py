import json
import codecs

def extract():
    with open('c:/Users/vmj07/OneDrive/Desktop/PySR/PySR_Formulas_Aquaculture.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with codecs.open('c:/Users/vmj07/OneDrive/Desktop/PySR/extracted_formulas_utf8.txt', 'w', 'utf-8') as out_f:
        for i, cell in enumerate(data.get('cells', [])):
            if cell.get('cell_type') == 'code':
                source = "".join(cell.get('source', []))
                out_f.write(f"=== Cell {i} ===\n")
                if "model_" in source:
                    out_f.write(f"Found model definition in source.\n")
                
                outputs = cell.get('outputs', [])
                for out in outputs:
                    if out.get('output_type') == 'stream':
                        text = "".join(out.get('text', []))
                        for line in text.split('\n'):
                            if "=>" in line or "Equation" in line or "Score" in line or "y =" in line or "model_" in line or line.strip().startswith("1") or line.strip().startswith("2") or "===" in line:
                                out_f.write(line + "\n")
                    elif out.get('output_type') in ['display_data', 'execute_result']:
                        data_dict = out.get('data', {})
                        if 'text/plain' in data_dict:
                            text = "".join(data_dict['text/plain'])
                            for line in text.split('\n'):
                                if "=>" in line or "Equation" in line or "Score" in line or "y =" in line or line.strip().startswith("1") or line.strip().startswith("2") or "===" in line:
                                    out_f.write(line + "\n")
                        if 'text/html' in data_dict:
                            text = "".join(data_dict['text/html'])
                            # Print a snippet of the HTML if it contains the word Equation
                            if "Equation" in text:
                                out_f.write("Found HTML table with equations\n")

if __name__ == "__main__":
    extract()
