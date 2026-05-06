import os
import glob

files = ['Home.py'] + glob.glob('pages/*.py')

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'load_css()' in content:
        continue
        
    lines = content.split('\n')
    insert_idx = -1
    
    # Try to find st.set_page_config
    for i, line in enumerate(lines):
        if 'st.set_page_config' in line:
            # Multi-line statement support: look for the closing parenthesis
            if ')' in line:
                insert_idx = i
            else:
                for j in range(i+1, len(lines)):
                    if ')' in lines[j]:
                        insert_idx = j
                        break
            break
            
    # If not found, find import streamlit as st
    if insert_idx == -1:
        for i, line in enumerate(lines):
            if 'import streamlit as st' in line:
                insert_idx = i
                break
                
    if insert_idx != -1:
        lines.insert(insert_idx + 1, 'from utils import load_css')
        lines.insert(insert_idx + 2, 'load_css()')
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Updated {filepath}")
    else:
        print(f"Could not find insertion point in {filepath}")
