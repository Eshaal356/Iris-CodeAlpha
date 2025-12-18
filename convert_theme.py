import re

# Read the file
with open('iris_dashboard.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all dark theme colors with light theme
replacements = [
    (r"paper_bgcolor='rgba\(10, 14, 39, 0\.8\)'", "paper_bgcolor='rgba(255, 255, 255, 0.9)'"),
    (r"plot_bgcolor='rgba\(255, 255, 255, 0\.03\)'", "plot_bgcolor='rgba(255, 255, 255, 1)'"),
    (r'font=\{\'color\': "#ffffff"\}', 'font={\'color\': "#2d3748"}'),
    (r"bgcolor='rgba\(10, 14, 39, 0\.5\)'", "bgcolor='rgba(255, 255, 255, 0.9)'"),
    (r"backgroundcolor='rgba\(255, 255, 255, 0\.02\)'", "backgroundcolor='rgba(240, 240, 240, 1)'"),
    (r"gridcolor='rgba\(255, 255, 255, 0\.1\)'", "gridcolor='rgba(200, 200, 200, 0.5)'"),
    (r'color: #ffffff', 'color: #2d3748'),
    (r'color: #a0aec0', 'color: #4a5568'),
    (r"textfont=dict\(color='#ffffff'", "textfont=dict(color='#2d3748'"),
    (r"line=dict\(color='rgba\(255, 255, 255, 0\.3\)'", "line=dict(color='rgba(0, 0, 0, 0.2)'"),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content)

# Write back
with open('iris_dashboard.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Theme conversion complete!")
