# JSON Schema Mapper Tool

## Overview

This Python script provides a convenient and interactive way to map JSON data files (`data.json`) into a 
specified schema format (`schema.json`). It's especially useful when your schema definitions frequently 
change, allowing flexible and reusable mapping.

Maps quantum inferno to the schema, maps key-value pairs.
Notifications can map to message body.

## How it Works

The script performs the following steps:

1. **Extract Keys**: It extracts keys from both your schema and data JSON files, including nested keys, 
presenting them clearly.

2. **Interactive Mapping**: It prompts you interactively in the terminal to map each key from your schema 
file to a corresponding key in your data file.

3. **Data Transformation**: Using the mappings you provided, it transforms the data file into the format 
specified by your schema.

4. **Mapping Persistence**: The mapping selections you make are saved in a file (`mapping.json` by 
default). Subsequent runs of the script can reuse this mapping file to bypass interactive prompts.

## How to Run

### First-time Run (Interactive Mode)
Run the script in your terminal using:

```bash
python script.py <schema.json> <data.json> <output.json>
```
- Replace `<schema.json>` with your schema file path.
- Replace `<data.json>` with your data file path.
- Replace `<output.json>` with the desired output file path.

The script will interactively prompt you to map keys.

### Subsequent Runs (Non-Interactive)
To reuse a previous mapping without interactive prompts, ensure the `mapping.json` file is in the current 
directory (or specify a custom mapping file):

```bash
python script.py <schema.json> <data.json> <output.json> [mapping.json]
```
- If the `mapping.json` file exists, the script automatically uses it, skipping the interactive prompt.

## Example Usage

```bash
python script.py schema.json data.json transformed_data.json
```

# Specific use:
```
$ python3 create_map_output_transformed_json.py sample_udl_payload.json sample_udl_payload.json output.json track-map.json
```

## Expected Outcome

- An output JSON file formatted according to the provided schema, populated with values mapped from your 
data file.
- A `mapping.json` file that saves your mapping choices for future use.

## Notes

- Ensure Python 3.x is installed.
- The script handles nested JSON structures and provides clear dot-notation for nested keys.

Enjoy seamless JSON transformations with flexibility and ease!
