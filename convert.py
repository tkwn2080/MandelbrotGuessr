import json

def convert_locations_to_js(input_file='mandelbrot_locations.json',
                           output_file='locations.js'):
    """Convert JSON locations to JavaScript file for the game"""

    # Load locations from JSON
    with open(input_file, 'r') as f:
        data = json.load(f)
        locations = data['locations']

    # Create JavaScript content
    js_content = "// Auto-generated Mandelbrot locations\n"
    js_content += "const MANDELBROT_LOCATIONS = [\n"

    for i, loc in enumerate(locations):
        js_content += f"  {{\n"
        js_content += f"    x: {loc['x']},\n"
        js_content += f"    y: {loc['y']},\n"
        js_content += f"    zoom: {loc['zoom']},\n"
        js_content += f"    difficulty: {loc['difficulty']},\n"
        js_content += f"    score: {loc['score']}\n"
        js_content += f"  }}"

        if i < len(locations) - 1:
            js_content += ","
        js_content += "\n"

    js_content += "];\n"

    # Save to JavaScript file
    with open(output_file, 'w') as f:
        f.write(js_content)

    print(f"Converted {len(locations)} locations to {output_file}")

if __name__ == "__main__":
    convert_locations_to_js()
