"""
Fine-tuning Script for CodeLlama on Code Review Data.
Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning on consumer GPUs.
"""
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def create_sample_dataset():
    """
    Create a sample code review dataset for fine-tuning.
    """
    data = [
        {
            "instruction": "Review the following Python code for issues:",
            "input": "def process_data(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result",
            "output": "Consider using list comprehension for better readability and performance: return [item * 2 for item in data]. This is more Pythonic and typically faster."
        },
        {
            "instruction": "Review the following Python code for security issues:",
            "input": "import os\npassword = 'admin123'\ncommand = 'login -u admin -p ' + password\nos.system(command)",
            "output": "CRITICAL SECURITY ISSUE: Hardcoded credentials detected. Never hardcode passwords in source code. Use environment variables or a secrets manager. Additionally, os.system() is dangerous - use subprocess.run() with proper argument handling to prevent command injection."
        },
        {
            "instruction": "Review the following Python code:",
            "input": "def get_user(id):\n    query = 'SELECT * FROM users WHERE id = ' + str(id)\n    cursor.execute(query)\n    return cursor.fetchone()",
            "output": "CRITICAL: SQL injection vulnerability! Never use string concatenation for SQL queries. Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (id,)). This prevents attackers from injecting malicious SQL."
        },
        {
            "instruction": "Review the following Python code for best practices:",
            "input": "class UserManager:\n    def __init__(self):\n        self.users = []\n    def add_user(self, user):\n        self.users.append(user)\n    def remove_user(self, user):\n        self.users.remove(user)\n    def find_user(self, user_id):\n        for u in self.users:\n            if u.id == user_id:\n                return u\n        return None",
            "output": "Consider using a dictionary instead of a list for O(1) lookup instead of O(n). Change to: self.users = {}. Then add_user becomes self.users[user.id] = user, and find_user becomes return self.users.get(user_id). This significantly improves performance for large user bases."
        },
        {
            "instruction": "Review the following Python code:",
            "input": "def calculate(x, y):\n    return x + y",
            "output": "Function lacks documentation. Add a docstring explaining parameters and return value. Also consider adding type hints: def calculate(x: int, y: int) -> int:. This improves code readability and enables better IDE support."
        },
        {
            "instruction": "Review the following Python code for performance:",
            "input": "result = []\nfor i in range(1000000):\n    result.append(i * 2)",
            "output": "Use list comprehension for better performance: result = [i * 2 for i in range(1000000)]. List comprehensions are optimized in CPython and typically 2-3x faster than equivalent for-loops. For very large datasets, consider using a generator expression: (i * 2 for i in range(1000000))."
        },
        {
            "instruction": "Review the following Python code:",
            "input": "if x == True:",
            "output": "Style issue: Use 'if x:' instead of 'if x == True:'. In Python, truthiness is checked directly. 'if x == True' is redundant and non-idiomatic. Similarly, use 'if not x:' instead of 'if x == False:'."
        },
        {
            "instruction": "Review the following Python code for error handling:",
            "input": "def divide(a, b):\n    return a / b",
            "output": "Missing error handling for division by zero. Add proper exception handling:\n\ndef divide(a, b):\n    try:\n        return a / b\n    except ZeroDivisionError:\n        raise ValueError('Cannot divide by zero')\n    except TypeError:\n        raise TypeError('Both arguments must be numbers')"
        },
        {
            "instruction": "Review the following Python code for security:",
            "input": "import pickle\ndata = pickle.loads(user_input)",
            "output": "CRITICAL SECURITY ISSUE: Unsafe deserialization! The pickle module can execute arbitrary code during deserialization. Never unpickle data from untrusted sources. Use json instead: import json; data = json.loads(user_input). If you must use pickle, implement strict input validation and use a whitelist of allowed types."
        },
        {
            "instruction": "Review the following Python code for design patterns:",
            "input": "def process():\n    data = fetch_data()\n    cleaned = clean_data(data)\n    analyzed = analyze_data(cleaned)\n    return analyzed",
            "output": "Consider using a pipeline pattern or method chaining for better readability and testability. For example:\n\nclass DataPipeline:\n    def __init__(self):\n        self.data = None\n    def fetch(self):\n        self.data = fetch_data()\n        return self\n    def clean(self):\n        self.data = clean_data(self.data)\n        return self\n    def analyze(self):\n        return analyze_data(self.data)\n\n# Usage: result = DataPipeline().fetch().clean().analyze()"
        },
        {
            "instruction": "Review the following Python code:",
            "input": "def fetch_data():\n    response = requests.get('https://api.example.com/data')\n    return response.json()",
            "output": "Missing error handling for network requests. HTTP requests can fail for many reasons. Add proper error handling:\n\ndef fetch_data():\n    try:\n        response = requests.get('https://api.example.com/data', timeout=30)\n        response.raise_for_status()\n        return response.json()\n    except requests.Timeout:\n        raise Exception('Request timed out')\n    except requests.HTTPError as e:\n        raise Exception('HTTP error: ' + str(e.response.status_code))\n    except requests.RequestException as e:\n        raise Exception('Request failed: ' + str(e))"
        },
        {
            "instruction": "Review the following Python code for memory efficiency:",
            "input": "def read_large_file(filename):\n    with open(filename, 'r') as f:\n        return f.read()",
            "output": "Reading entire large files into memory can cause memory issues. For large files, use a generator to process line by line:\n\ndef read_large_file(filename):\n    with open(filename, 'r') as f:\n        for line in f:\n            yield line.strip()\n\n# Usage: for line in read_large_file('huge_file.txt'): process(line)"
        },
        {
            "instruction": "Review the following Python code for concurrency:",
            "input": "import time\n\ndef fetch_all(urls):\n    results = []\n    for url in urls:\n        results.append(requests.get(url).json())\n    return results",
            "output": "Sequential HTTP requests are slow. Use asyncio or concurrent.futures for parallel requests:\n\nimport asyncio\nimport aiohttp\n\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_one(session, url) for url in urls]\n        return await asyncio.gather(*tasks)\n\nasync def fetch_one(session, url):\n    async with session.get(url) as response:\n        return await response.json()"
        },
        {
            "instruction": "Review the following Python code for API design:",
            "input": "class Database:\n    def __init__(self):\n        self.connection = None\n    def connect(self, host, port, username, password):\n        self.connection = psycopg2.connect(...)\n    def query(self, sql):\n        cursor = self.connection.cursor()\n        cursor.execute(sql)\n        return cursor.fetchall()",
            "output": "Good structure, but consider these improvements:\n1. Use context managers for connections: 'with self.connection.cursor() as cursor:'\n2. Add connection pooling with psycopg2.pool or SQLAlchemy\n3. Use parameterized queries to prevent SQL injection\n4. Add connection health checks and automatic reconnection\n5. Consider using an ORM like SQLAlchemy for complex queries"
        },
        {
            "instruction": "Review the following Python code for logging:",
            "input": "def process_order(order):\n    print('Processing order ' + str(order.id))\n    # ... processing logic ...\n    print('Order processed successfully')",
            "output": "Use proper logging instead of print statements. Print output goes to stdout and is hard to filter and manage. Use the logging module:\n\nimport logging\nlogger = logging.getLogger(__name__)\n\ndef process_order(order):\n    logger.info('Processing order %s', order.id)\n    try:\n        # ... processing logic ...\n        logger.info('Order %s processed successfully', order.id)\n    except Exception as e:\n        logger.error('Failed to process order %s: %s', order.id, str(e))\n        raise"
        }
    ]
    
    return Dataset.from_list(data)


def format_instruction(example):
    """
    Format dataset examples into instruction-following format.
    Uses CodeLlama's chat template.
    """
    instruction = example["instruction"]
    input_text = example["input"]
    output = example["output"]
    
    # CodeLlama Instruct format
    prompt = "[INST] " + instruction + "\n\nCode:\n```python\n" + input_text + "\n```\n[/INST]\n\n" + output
    
    return {"text": prompt}


def main():
    """Main fine-tuning pipeline."""
    print("=" * 60)
    print("Code Review Assistant - Model Fine-tuning")
    print("=" * 60)
    
    # Configuration
    MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
    OUTPUT_DIR = "./models/codellama-finetuned"
    
    print("\n1. Loading base model: " + MODEL_NAME)
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    print("   Model loaded. Device map: " + str(model.hf_device_map))
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    print("\n2. Configuring LoRA adapters")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "v_proj", 
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare dataset
    print("\n3. Preparing dataset")
    dataset = create_sample_dataset()
    formatted_dataset = dataset.map(format_instruction)
    
    # Split into train/test
    split_dataset = formatted_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    print("   Training samples: " + str(len(train_dataset)))
    print("   Evaluation samples: " + str(len(eval_dataset)))
    
    # Training arguments
    print("\n4. Starting training")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        optim="paged_adamw_8bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="none",
        evaluation_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
    )
    
    # Train
    print("\n   Training in progress...")
    trainer.train()
    
    # Save model
    print("\n5. Saving fine-tuned model to " + OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save adapter config
    model.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print("Model saved to: " + OUTPUT_DIR)
    print("=" * 60)
    
    # Test the model
    print("\n6. Testing model with sample input")
    test_prompt = "[INST] Review the following Python code for issues:\n\nCode:\n```python\ndef process(items):\n    result = []\n    for i in items:\n        result.append(i + 1)\n    return result\n```\n[/INST]\n"
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\nTest output:")
    print(response[len(test_prompt):])


if __name__ == "__main__":
    main()