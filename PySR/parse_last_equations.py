import json

def extract():
    with open('c:/Users/vmj07/OneDrive/Desktop/PySR/PySR_Formulas_Aquaculture.ipynb', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    current_model = ""
    model_tables = {}

    for i, cell in enumerate(data.get('cells', [])):
        if cell.get('cell_type') == 'code':
            source = "".join(cell.get('source', []))
            for line in source.split('\n'):
                if "TRAINING MODEL" in line:
                    current_model = line.strip().replace("=", "").replace("-", "").strip()
            
            outputs = cell.get('outputs', [])
            for out in outputs:
                text = ""
                if out.get('output_type') == 'stream':
                    text = "".join(out.get('text', []))
                elif out.get('output_type') in ['display_data', 'execute_result']:
                    data_dict = out.get('data', {})
                    if 'text/plain' in data_dict:
                        text = "".join(data_dict['text/plain'])
                
                if "Complexity" in text and "Loss" in text and "Equation" in text:
                    tables = text.split("Complexity  Loss       Score      Equation")
                    if tables:
                        last_table = tables[-1].strip()
                        lines = [l for l in last_table.split('\n') if l.strip()]
                        # The equation might span multiple lines if it wraps. Let's get the whole text
                        if current_model:
                            model_tables[current_model] = lines

    for model, lines in model_tables.items():
        print(f"Model: {model}")
        print(f"Most accurate formula (last row):")
        for line in lines[-3:]:
            print(line)
        print("-" * 50)

if __name__ == "__main__":
    extract()
