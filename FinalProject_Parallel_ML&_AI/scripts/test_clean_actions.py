"""
Test that clean prompts produce valid actions - NO training
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "gnupg", "software-properties-common", "ca-certificates", "curl")
    .run_commands(
        "mkdir -p /etc/apt/keyrings",
        "wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc",
        "echo 'deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb bookworm main' | tee /etc/apt/sources.list.d/adoptium.list",
        "apt-get update",
        "apt-get install -y temurin-21-jdk",
    )
    .pip_install(
        "numpy>=1.25.0", "torch>=2.0.0", "transformers>=4.35.0", "datasets>=2.14.0",
        "tqdm>=4.66.0", "pyyaml>=6.0", "accelerate>=0.24.0",
        "faiss-cpu>=1.7.0", "pyserini", "gym==0.24.0", "spacy>=3.6.0",
        "flask==2.1.2", "Werkzeug==2.0.3", "beautifulsoup4", "rank-bm25",
        "nltk", "cleantext", "requests", "scikit-learn", "pandas",
        "selenium", "gdown", "rich", "thefuzz", "python-Levenshtein",
    )
    .run_commands("python -m spacy download en_core_web_sm")
)

app = modal.App("test-clean-actions", image=image)

@app.function(gpu="H100", timeout=600)
def test_actions():
    import subprocess, sys, os, re
    
    # Setup WebShop
    if not os.path.exists("/root/WebShop"):
        subprocess.run(["git", "clone", "https://github.com/princeton-nlp/WebShop.git", "/root/WebShop"], check=True)
    
    os.chdir("/root/WebShop")
    os.makedirs("data", exist_ok=True)
    
    for file_id, path in [
        ("1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib", "data/items_shuffle_1000.json"),
        ("1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu", "data/items_ins_v2_1000.json"),
        ("14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O", "data/items_human_ins.json"),
    ]:
        if not os.path.exists(path):
            subprocess.run([sys.executable, "-m", "gdown", file_id, "-O", path], check=False)
    
    # Build index
    os.makedirs("search_engine", exist_ok=True)
    sys.path.insert(0, "/root/WebShop")
    from web_agent_site.utils import DEFAULT_FILE_PATH
    from web_agent_site.engine.engine import load_products
    import json
    
    all_products, *_ = load_products(filepath=DEFAULT_FILE_PATH, num_products=100, human_goals=True)
    os.makedirs("search_engine/resources_100", exist_ok=True)
    
    docs = []
    for p in all_products:
        option_texts = []
        for name, contents in p.get('options', {}).items():
            option_texts.append(f'{name}: {", ".join(contents)}')
        doc = {
            'id': p['asin'],
            'contents': ' '.join([
                p['Title'], p['Description'],
                p['BulletPoints'][0] if p.get('BulletPoints') else '',
                ', and '.join(option_texts),
            ]).lower()
        }
        docs.append(doc)
    
    with open('search_engine/resources_100/documents.jsonl', 'w') as f:
        for doc in docs:
            f.write(json.dumps(doc) + '\n')
    
    subprocess.run([
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection", "--input", "search_engine/resources_100",
        "--index", "search_engine/indexes_100", "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1", "--storePositions", "--storeDocvectors", "--storeRaw"
    ], check=True, cwd="/root/WebShop", capture_output=True)
    
    os.environ['PYTHONPATH'] = "/root/WebShop"
    
    # Load model
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from web_agent_site.envs import WebAgentTextEnv
    
    print("📦 Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        torch_dtype=torch.float16,
        device_map='cuda',
        trust_remote_code=True
    )
    
    env = WebAgentTextEnv(observation_mode='text', num_products=100)
    
    # Load tasks
    with open('/root/WebShop/data/items_ins_v2_1000.json') as f:
        tasks = list(json.load(f).values())[:20]
    
    print("\n" + "="*60)
    print("🧪 TESTING ACTION GENERATION (NO TRAINING)")
    print("="*60 + "\n")
    
    valid_count = 0
    total_count = 0
    
    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}/20")
        obs = env.reset(session=task)
        
        for turn in range(3):
            prompt = f"""You are a WebShop agent. Output ONLY actions.

VALID ACTIONS:
search[query]
click[B0XXXXXXX]
buy now
back

Task: {task}
State: {obs}

Action:"""
            
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=30, temperature=0.7,
                                   do_sample=True, pad_token_id=tokenizer.eos_token_id)
            
            resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
            # Check if valid action
            match = re.search(r'search\[([^\]]+)\]', resp, re.IGNORECASE)
            if match:
                action = f"search[{match.group(1)[:50]}]"
                print(f"  ✓ Turn {turn}: {action[:40]}")
                valid_count += 1
            else:
                print(f"  ❌ Turn {turn}: INVALID ({resp[:40]})")
            
            total_count += 1
            obs, _, done, _ = env.step(action if match else 'search[products]')
            if done: break
    
    print("\n" + "="*60)
    print(f"✅ RESULTS: {valid_count}/{total_count} valid actions ({100*valid_count/total_count:.1f}%)")
    print("="*60)
    
    return {'valid': valid_count, 'total': total_count}

@app.local_entrypoint()
def main():
    result = test_actions.remote()
    print(f"\n✅ Test complete: {result}")