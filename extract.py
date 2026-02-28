import docx

for doc_name in ["PRD.docx", "design.docx", "tech_stack.docx"]:
    doc = docx.Document(doc_name)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    
    with open(doc_name.replace('.docx', '.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(text))
