import os
import json
import time
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


MODEL_CONFIGS = {
    'unixcoder': {
        'name': 'microsoft/unixcoder-base',
        'max_length': 512,
        'use_sentence_transformer': False
    },
    'codexembed': {
        'name': 'Salesforce/SFR-Embedding-Code-400M_R',
        'max_length': 512,
        'use_sentence_transformer': True
    },
    'graphcodebert': {
        'name': 'microsoft/graphcodebert-base',
        'max_length': 512,
        'use_sentence_transformer': False
    },
    'codebert': {
        'name': 'microsoft/codebert-base',
        'max_length': 512,
        'use_sentence_transformer': False
    },
    'starencoder': {
        'name': 'bigcode/starencoder',
        'max_length': 1024,
        'use_sentence_transformer': False
    }
}


def load_test_cases(test_file_path):
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"Test file not found: {test_file_path}")

    tests = []
    with open(test_file_path, 'r') as f:
        for line in f:
            test_code = line.strip()
            if test_code:
                tests.append(test_code)

    if len(tests) == 0:
        raise ValueError(f"No test cases found in {test_file_path}")

    return tests


def generate_embeddings_transformer(tests, model_name, tokenizer, model, device, batch_size=16, max_length=512):
    """Generate embeddings using transformers AutoModel (for graphcodebert, unixcoder, codebert, starencoder)"""
    embeddings = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(tests), batch_size), desc=f"Generating embeddings"):
            batch_tests = tests[i:i+batch_size]

            inputs = tokenizer(
                batch_tests,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)

            outputs = model(**inputs)

            if hasattr(outputs, 'last_hidden_state'):
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            elif hasattr(outputs, 'pooler_output'):
                batch_embeddings = outputs.pooler_output.cpu().numpy()
            else:
                batch_embeddings = outputs[0].mean(dim=1).cpu().numpy()

            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def generate_embeddings_sentence_transformer(tests, model, batch_size=16, normalize=True):
    """Generate embeddings using SentenceTransformer (for SFR model)"""
    embeddings = model.encode(
        tests,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,
        convert_to_numpy=True
    )

    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Generate test embeddings for multiple models')
    parser.add_argument('--project', type=str, required=True, help='Project name (e.g., time_v0)')
    parser.add_argument('--models', type=str, nargs='+', default=list(MODEL_CONFIGS.keys()),
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Models to use for embedding generation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for embedding generation')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if embeddings exist')

    args = parser.parse_args()

    input_dir = f'input/{args.project}'
    test_file = f'{input_dir}/{args.project.split("_")[0]}-bbox.txt'
    embeddings_dir = f'{input_dir}/embeddings'

    os.makedirs(embeddings_dir, exist_ok=True)

    print(f"Loading test cases from {test_file}")
    tests = load_test_cases(test_file)
    print(f"Loaded {len(tests)} test cases")

    for model_key in args.models:
        model_config = MODEL_CONFIGS[model_key]
        model_name = model_config['name']
        max_length = model_config['max_length']
        use_sentence_transformer = model_config.get('use_sentence_transformer', False)

        embeddings_file = f'{embeddings_dir}/{model_key}_test_embeddings.npy'
        timing_file = f'{embeddings_dir}/{model_key}_timing.json'

        if os.path.exists(embeddings_file) and not args.force:
            print(f"\nEmbeddings for {model_key} already exist at {embeddings_file}")
            print(f"Loading existing embeddings...")
            embeddings = np.load(embeddings_file)
            print(f"Loaded embeddings with shape {embeddings.shape}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        method_name = 'SentenceTransformer' if use_sentence_transformer else 'Transformers AutoModel'
        print(f"Method: {method_name}")
        print(f"{'='*60}")

        load_start = time.time()

        if use_sentence_transformer:
            # Use SentenceTransformer for SFR model
            try:
                print(f"Loading SentenceTransformer model...")
                model = SentenceTransformer(model_name, trust_remote_code=True)
                load_time = time.time() - load_start
                print(f"Model loaded successfully in {load_time:.2f} seconds")
                print(f"Max sequence length: {model.max_seq_length}")
            except Exception as e:
                print(f"ERROR: Failed to load SentenceTransformer model {model_name}: {e}")
                raise

            embed_start = time.time()
            embeddings = generate_embeddings_sentence_transformer(
                tests, model, args.batch_size, normalize=True
            )
            embed_time = time.time() - embed_start

        else:
            # Use transformers AutoModel for other models
            try:
                print(f"Loading tokenizer and model...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

                # Set padding token if not already set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    if tokenizer.pad_token is None:
                        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

                model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                model.to(args.device)
                model.eval()
                load_time = time.time() - load_start
                print(f"Model loaded successfully in {load_time:.2f} seconds")
            except Exception as e:
                print(f"ERROR: Failed to load model {model_name}: {e}")
                raise

            embed_start = time.time()
            embeddings = generate_embeddings_transformer(
                tests, model_name, tokenizer, model,
                args.device, args.batch_size, max_length
            )
            embed_time = time.time() - embed_start

        total_time = load_time + embed_time

        print(f"\nSaving embeddings to {embeddings_file}")
        np.save(embeddings_file, embeddings)
        print(f"Embeddings shape: {embeddings.shape}")

        timing_data = {
            'model': model_name,
            'num_tests': len(tests),
            'embedding_dim': embeddings.shape[1],
            'load_time': load_time,
            'embedding_time': embed_time,
            'total_time': total_time,
            'device': args.device,
            'batch_size': args.batch_size,
            'method': method_name
        }

        print(f"Saving timing data to {timing_file}")
        with open(timing_file, 'w') as f:
            json.dump(timing_data, f, indent=2)

        print(f"\nTiming Summary for {model_key}:")
        print(f"  Model loading time: {load_time:.2f}s")
        print(f"  Embedding generation time: {embed_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")

        del model
        if not use_sentence_transformer:
            del tokenizer
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Embedding generation completed for all models!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()