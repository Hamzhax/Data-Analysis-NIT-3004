#!/usr/bin/env python3

with open('app.py', 'r') as f:
    lines = f.readlines()

# Find the problematic lines
in_pdf_func = False
try_found = False
try_line = None
except_line = None

for i, line in enumerate(lines):
    if 'def generate_comprehensive_pdf_report(' in line:
        in_pdf_func = True
    elif in_pdf_func and line.strip().startswith('try:'):
        try_found = True
        try_line = i
    elif in_pdf_func and try_found and line.strip().startswith('except Exception as e:') and not line.startswith('        '):
        except_line = i
        break

if try_line is not None and except_line is not None:
    print(f"Fixing indentation from line {try_line+1} to {except_line}")
    
    # Fix the indentation - everything between try and except should be indented by 4 more spaces
    for i in range(try_line + 1, except_line):
        if lines[i].strip():  # Only process non-empty lines
            # Check current indentation level
            current_indent = len(lines[i]) - len(lines[i].lstrip())
            # Add 4 spaces if not properly indented for try block
            if not lines[i].startswith('        '):  # Should be at least 8 spaces (function + try)
                if lines[i].startswith('    '):  # Currently has 4 spaces, add 4 more
                    lines[i] = '    ' + lines[i]
                elif not lines[i].startswith(' '):  # No indentation, add 8 spaces
                    lines[i] = '        ' + lines[i]
    
    # Write the fixed file
    with open('app.py', 'w') as f:
        f.writelines(lines)
    
    print("Indentation fixed!")
else:
    print("Could not find the problematic section")
