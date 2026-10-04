import json
import codecs

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
                    current_model = line.strip().replace("=", "").strip()
            
            outputs = cell.get('outputs', [])
            for out in outputs:
                text = ""
                if out.get('output_type') == 'stream':
                    text = "".join(out.get('text', []))
                elif out.get('output_type') in ['display_data', 'execute_result']:
                    data_dict = out.get('data', {})
                    if 'text/plain' in data_dict:
                        text = "".join(data_dict['text/plain'])
                
                if "Complexity  Loss       Score      Equation" in text:
                    tables = text.split("Complexity  Loss       Score      Equation")
                    if len(tables) > 1:
                        last_table = tables[-1].strip()
                        lines = [l for l in last_table.split('\n') if l.strip()]
                        if current_model:
                            model_tables[current_model] = lines

    with codecs.open('c:/Users/vmj07/OneDrive/Desktop/PySR/parsed_best_formulas.txt', 'w', 'utf-8') as out_f:
        for model, lines in model_tables.items():
            out_f.write(f"Model: {model}\n")
            out_f.write("Most accurate formula (last row):\n")
            # Usually the last equation takes up multiple lines if it is long.
            # So let's write out the last 5 lines and the user can see it.
            for line in lines[-5:]:
                out_f.write(line + "\n")
            out_f.write("-" * 50 + "\n")

if __name__ == "__main__":
    extract()
