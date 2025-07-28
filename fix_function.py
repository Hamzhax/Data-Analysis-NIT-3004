#!/usr/bin/env python3

with open('app.py', 'r') as f:
    content = f.read()

# Find the function and fix it completely by rebuilding the entire function with proper indentation

start_marker = "def generate_comprehensive_pdf_report(bundle, ai, filename):"
end_marker = "finally:\n        # Clean up chart images to free memory"

start_idx = content.find(start_marker)
if start_idx == -1:
    print("Could not find function start")
    exit(1)

# Find the end by looking for the finally block
end_idx = content.find(end_marker, start_idx)
if end_idx == -1:
    print("Could not find function end")
    exit(1)

# Find the actual end of the finally block
finally_end = content.find("\n\n@app", end_idx)
if finally_end == -1:
    finally_end = content.find("\n@app", end_idx)

if finally_end == -1:
    print("Could not find function end marker")
    exit(1)

function_content = content[start_idx:finally_end]
print(f"Found function content length: {len(function_content)}")

# Now we rebuild the function with proper indentation
# The problem is that everything after "try:" should be indented 4 more spaces

lines = function_content.split('\n')
fixed_lines = []
in_try_block = False

for line in lines:
    if 'try:' in line and line.strip() == 'try:':
        in_try_block = True
        fixed_lines.append(line)
    elif in_try_block and (line.strip().startswith('except ') or line.strip().startswith('finally:')):
        in_try_block = False
        fixed_lines.append(line)
    elif in_try_block:
        # This line should be indented to be inside the try block
        # Check if it already has enough indentation
        if line.strip():  # Only process non-empty lines
            current_indent = len(line) - len(line.lstrip())
            if current_indent < 8:  # Should have at least 8 spaces (4 for function + 4 for try)
                # Add 4 more spaces to whatever it currently has
                fixed_lines.append('    ' + line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    else:
        fixed_lines.append(line)

fixed_function = '\n'.join(fixed_lines)

# Replace the function in the original content
new_content = content[:start_idx] + fixed_function + content[finally_end:]

with open('app.py', 'w') as f:
    f.write(new_content)

print("Function indentation fixed!")
