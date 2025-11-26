"""
All-in-One Modal Script for WebShop A*-PO Training
Simple structure following modal_train_ragen.py pattern
"""
import modal

# Build image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", "wget", "gnupg", "software-properties-common",
        "ca-certificates", "curl"
    )
    .run_commands(
        # Install Java 21 for WebShop
        "mkdir -p /etc/apt/keyrings",
        "wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc",
        "echo 'deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb bookworm main' | tee /etc/apt/sources.list.d/adoptium.list",
        "apt-get update",
        "apt-get install -y temurin-21-jdk",
    )
    .pip_install(
        # Core ML
        "numpy>=1.25.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "tqdm>=4.66.0",
        "accelerate>=0.24.0",

        # WebShop dependencies (same as modal_train_ragen.py)
        "faiss-cpu>=1.7.0",
        "pyserini",
        "gym==0.24.0",
        "spacy>=3.6.0",
        "flask==2.1.2",
        "Werkzeug==2.0.3",
        "beautifulsoup4",
        "rank-bm25",
        "nltk",
        "cleantext",
        "requests",
        "scikit-learn",
        "pandas",
        "selenium",
        "gdown",
        "rich",
        "thefuzz",
        "python-Levenshtein",
    )
    .run_commands(
        "python -m spacy download en_core_web_sm"
    )
)

app = modal.App("webshop-apo", image=image)
volume = modal.Volume.from_name("webshop-apo-outputs", create_if_missing=True)


@app.function(
    gpu="H100",
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
)
def train(code_tar: bytes, args_dict: dict):
    """Run A*-PO training on WebShop"""
    import subprocess
    import sys
    import os
    import tarfile
    import io
    import json
    from pathlib import Path

    print("="*60, flush=True)
    print("🤖 A*-PO WebShop Training on H100", flush=True)
    print("="*60, flush=True)

    # Setup working directory
    work_dir = "/root/work"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # Extract code
    print("\n📦 Extracting code...", flush=True)
    tar = tarfile.open(fileobj=io.BytesIO(code_tar))
    tar.extractall()
    tar.close()

    # Setup WebShop
    print("\n📥 Setting up WebShop...", flush=True)

    if not os.path.exists("/root/WebShop"):
        subprocess.run([
            "git", "clone",
            "https://github.com/princeton-nlp/WebShop.git",
            "/root/WebShop"
        ], check=True)

    os.chdir("/root/WebShop")
    os.makedirs("data", exist_ok=True)

    # Download WebShop data
    print("\n📥 Downloading WebShop data...", flush=True)

    data_files = [
        ("1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib", "data/items_shuffle_1000.json"),
        ("1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu", "data/items_ins_v2_1000.json"),
        ("14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O", "data/items_human_ins.json"),
    ]

    for file_id, output_path in data_files:
        if not os.path.exists(output_path):
            subprocess.run([
                sys.executable, "-m", "gdown", file_id, "-O", output_path
            ], check=False)

    # Setup search indexes
    print("\n📥 Setting up search indexes...", flush=True)

    packaged_indexes = os.path.join(work_dir, "webshop_indexes")

    if os.path.exists(packaged_indexes):
        print("✓ Using packaged pre-built indexes", flush=True)
        os.makedirs("search_engine", exist_ok=True)
        subprocess.run(["cp", "-r", packaged_indexes, "search_engine/indexes_100"], check=True)
    else:
        print("⚠️ Building indexes from scratch...", flush=True)

        os.makedirs("search_engine", exist_ok=True)

        # Import WebShop
        sys.path.insert(0, "/root/WebShop")
        from web_agent_site.utils import DEFAULT_FILE_PATH
        from web_agent_site.engine.engine import load_products

        print("  Loading products...", flush=True)
        all_products, *_ = load_products(
            filepath=DEFAULT_FILE_PATH,
            num_products=100,
            human_goals=True
        )

        print("  Creating index documents...", flush=True)
        os.makedirs("search_engine/resources_100", exist_ok=True)

        docs = []
        for p in all_products:
            option_texts = []
            options = p.get('options', {})
            for option_name, option_contents in options.items():
                option_contents_text = ', '.join(option_contents)
                option_texts.append(f'{option_name}: {option_contents_text}')
            option_text = ', and '.join(option_texts)

            doc = {
                'id': p['asin'],
                'contents': ' '.join([
                    p['Title'],
                    p['Description'],
                    p['BulletPoints'][0] if p.get('BulletPoints') else '',
                    option_text,
                ]).lower()
            }
            docs.append(doc)

        with open('search_engine/resources_100/documents.jsonl', 'w') as f:
            for doc in docs:
                f.write(json.dumps(doc) + '\n')

        print("  Building Lucene index...", flush=True)
        subprocess.run([
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", "search_engine/resources_100",
            "--index", "search_engine/indexes_100",
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ], check=True, cwd="/root/WebShop")

        print("✓ Indexes built!", flush=True)

    # Verify indexes
    index_path = "/root/WebShop/search_engine/indexes_100"
    if not os.path.exists(index_path):
        print(f"❌ Index not found at {index_path}", flush=True)
        return {"status": "failed", "error": "Index not found"}

    print(f"✓ Index verified at {index_path}", flush=True)

    # Set WebShop path
    sys.path.insert(0, "/root/WebShop")
    os.environ['WEBSHOP_PATH'] = "/root/WebShop"
    os.environ['PYTHONPATH'] = "/root/WebShop:" + os.environ.get('PYTHONPATH', '')

    # Back to work directory
    os.chdir(work_dir)

    # Verify environment
    print("\n🔍 Verifying environment...", flush=True)
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    print(f"Java: {result.stderr.split('version')[1].split()[0] if 'version' in result.stderr else 'unknown'}", flush=True)

    for mod in ["torch", "transformers", "spacy", "pyserini"]:
        try:
            m = __import__(mod)
            print(f"✓ {mod}: {getattr(m, '__version__', 'OK')}", flush=True)
        except Exception as e:
            print(f"❌ {mod}: {e}", flush=True)
            return {"status": "failed", "error": f"Failed to import {mod}"}

    print("✓ Environment ready!", flush=True)

    # Start training
    print("\n" + "="*60, flush=True)
    print("🚀 STARTING TRAINING", flush=True)
    print("="*60, flush=True)

    # Build command line arguments
    cmd = [sys.executable, "-u", "train.py"]

    # Add arguments
    for key, value in args_dict.items():
        if value is not None:
            cmd.append(f"--{key.replace('_', '-')}")
            cmd.append(str(value))

    # Set output directory to volume
    cmd.extend(["--output-dir", "/outputs"])

    print(f"Command: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, cwd=work_dir)

    if result.returncode != 0:
        return {"status": "failed", "exit_code": result.returncode}

    print("\n🎉 SUCCESS!", flush=True)
    volume.commit()

    return {"status": "completed", "exit_code": 0}


@app.local_entrypoint()
def main(
    # Model
    model: str = "Qwen/Qwen2.5-3B-Instruct",

    # Training
    num_steps: int = 100,
    batch_size: int = 2,

    # A*-PO
    beta: float = 0.5,
    v_star_samples: int = 8,
    learning_rate: float = 5e-6,
    kl_coef: float = 0.02,

    # Environment
    num_products: int = 100,
    max_episode_steps: int = 15,

    # Evaluation
    eval_frequency: int = 20,
    eval_episodes: int = 10,

    # Checkpointing
    save_frequency: int = 50,

    # Misc
    seed: int = 42,
):
    """Deploy A*-PO training to Modal"""
    import tarfile
    import io
    from pathlib import Path

    print("\n📦 PACKAGING CODE")

    if not Path("webshop_apo").exists():
        print("❌ Run from week_06/ directory!")
        return

    # Package code
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        # Add webshop_apo module
        for file in ['apo_trainer.py', 'policy.py', 'webshop_env.py', 'train.py']:
            tar.add(f'webshop_apo/{file}', arcname=file)

        # Add pre-built indexes if available
        idx = Path("../WebShop/search_engine/indexes_100")
        if idx.exists():
            print("✓ Including pre-built indexes (saves 5-10 minutes)", flush=True)
            tar.add(str(idx), arcname='webshop_indexes')

    code_tar = tar_buffer.getvalue()
    print(f"✓ Package size: {len(code_tar)/1024/1024:.1f} MB")

    # Collect arguments
    args_dict = {
        'model': model,
        'num_steps': num_steps,
        'batch_size': batch_size,
        'beta': beta,
        'v_star_samples': v_star_samples,
        'learning_rate': learning_rate,
        'kl_coef': kl_coef,
        'num_products': num_products,
        'max_episode_steps': max_episode_steps,
        'eval_frequency': eval_frequency,
        'eval_episodes': eval_episodes,
        'save_frequency': save_frequency,
        'seed': seed,
    }

    print("\n🚀 DEPLOYING TO MODAL...")
    print(f"  Model: {model}")
    print(f"  Training steps: {num_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  V* samples: {v_star_samples}")
    print(f"  Products: {num_products}")

    result = train.remote(code_tar, args_dict)

    if result.get("status") == "completed":
        print("\n🎉 SUCCESS!")
        print("Download results:")
        print("  modal volume get webshop-apo-outputs /outputs ./webshop_results")
    else:
        print(f"\n⚠️ Failed: {result}")


if __name__ == "__main__":
    main()
