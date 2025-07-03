import json
import sys
import os

# Load JSON data from file
def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Save JSON data to file
def save_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=2)

# Recursively extract keys from JSON
def extract_keys(obj, prefix=''):
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.extend(extract_keys(v, full_key))
    elif isinstance(obj, list) and obj:
        keys.extend(extract_keys(obj[0], prefix))
    else:
        keys.append(prefix)
    return keys

# Retrieve nested value using dot-notation keys
def get_nested_value(data, key):
    keys = key.split('.')
    for k in keys:
        if isinstance(data, list):
            data = data[0]
        data = data.get(k, {})
    return data if data else None

# Interactive mapping of keys from schema to data
def interactive_mapping(schema_keys, data_keys):
    mapping = {}
    print("\nMap schema keys to data keys:")
    for s_key in schema_keys:
        print(f"\nSchema key: {s_key}")
        print("Select the corresponding data key:")
        for idx, d_key in enumerate(data_keys):
            print(f"[{idx}] {d_key}")
        choice = input("Enter choice number (or leave blank to skip): ")
        if choice.isdigit() and int(choice) < len(data_keys):
            mapping[s_key] = data_keys[int(choice)]
        else:
            mapping[s_key] = None
    return mapping

# Transform data according to mapping
def transform_data(schema_keys, data, mapping):
    transformed = {}
    for s_key in schema_keys:
        d_key = mapping.get(s_key)
        transformed[s_key] = get_nested_value(data, d_key) if d_key else None
    return transformed

def main(schema_path, data_path, output_path, mapkey_path="mapping.json"):
    schema = load_json(schema_path)
    data = load_json(data_path)

    schema_keys = extract_keys(schema)
    data_keys = extract_keys(data)

    if os.path.exists(mapkey_path):
        mapping = load_json(mapkey_path)
        print("Using existing mapping file.")
    else:
        mapping = interactive_mapping(schema_keys, data_keys)
        save_json(mapkey_path, mapping)
        print(f"Mapping saved to {mapkey_path}")

    transformed = transform_data(schema_keys, data, mapping)
    save_json(output_path, transformed)

    print(f"\nTransformed JSON saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python script.py <schema.json> <data.json> <output.json> [<mapping.json>]")
        sys.exit(1)

    schema_path, data_path, output_path = sys.argv[1:4]
    mapkey_path = sys.argv[4] if len(sys.argv) == 5 else "mapping.json"
    main(schema_path, data_path, output_path, mapkey_path)

