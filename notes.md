# Register venv as jupyter kernel

[-] python -m ipykernel install --user --name=rag_env --display-name "Python (rag_env)"

[-] jupyter notebook

[-] Kernel → Change Kernel → Python (rag_env)

# Model training results:
Fine-tuned MRR: 0.9260174868461777 


Fine-tuned Recall: {'Recall@1': 0.8997214484679665, 'Recall@5': 0.9623955431754875, 'Recall@10': 0.9693593314763231}

# No-training results:
Baseline MRR: 0.8815780165362339 


Baseline Recall: {'Recall@1': 0.8440111420612814, 'Recall@5': 0.9275766016713092, 'Recall@10': 0.9484679665738162}