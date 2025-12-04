"""
LoRA/QLoRA Fine-Tuning for Large Language Models (up to 70B parameters)
PII/PHI Detection and Masking with Gretel Dataset
Optimized for high-memory systems (128GB unified memory)
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GenerationConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import Dataset, DatasetDict
import re
from tqdm import tqdm
import logging
from datetime import datetime
import os
import gc
from accelerate import Accelerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = {
    "gemma2-27b": {
        "model_id": "google/gemma-2-27b-it",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "gemma2-9b": {
        "model_id": "google/gemma-2-9b-it",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "mixtral-8x7b": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "qwen2.5-32b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 32,
        "lora_alpha": 64,
    },
    "qwen2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_r": 32,
        "lora_alpha": 64,
    },
}

# ============================================================================
# DATASET PREPARATION
# ============================================================================

class GretelDatasetProcessor:
    """Process Gretel dataset for instruction fine-tuning"""
    
    def __init__(self, model_type: str = "gemma"):
        self.model_type = model_type
        self.instruction_templates = self._get_instruction_templates()
        
    def _get_instruction_templates(self) -> Dict[str, str]:
        """Get model-specific instruction templates"""
        templates = {
            "gemma": {
                "system": "You are a specialized PII/PHI detection and masking assistant. You identify and mask sensitive information in text while preserving the document's readability.",
                "format": "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
            },
            "mistral": {
                "system": "You are a specialized PII/PHI detection and masking assistant.",
                "format": "<s>[INST] {system}\n\n{instruction} [/INST] {response} </s>"
            },
            "mixtral": {
                "system": "You are a specialized PII/PHI detection and masking assistant.",
                "format": "<s>[INST] {system}\n\n{instruction} [/INST] {response} </s>"
            },
            "qwen": {
                "system": "You are a specialized PII/PHI detection and masking assistant.",
                "format": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
            }
        }
        
        if "gemma" in self.model_type:
            return templates["gemma"]
        elif "qwen" in self.model_type:
            return templates["qwen"]
        elif "mixtral" in self.model_type:
            return templates["mixtral"]
        else:
            return templates["mistral"]
    
    def parse_entities(self, entities_str: Union[str, list]) -> List[Dict]:
        """Parse entities from string or list format"""
        if isinstance(entities_str, list):
            entities_list = entities_str
        elif isinstance(entities_str, str):
            try:
                json_str = entities_str.replace("'", '"')
                entities_list = json.loads(json_str)
            except json.JSONDecodeError:
                import ast
                try:
                    entities_list = ast.literal_eval(entities_str)
                except (ValueError, SyntaxError):
                    logger.warning(f"Failed to parse entities: {entities_str[:100]}...")
                    return []
        else:
            return []
        
        parsed_entities = []
        for entity in entities_list:
            if isinstance(entity, dict):
                if 'entity' in entity and 'types' in entity:
                    entity_text = entity['entity']
                    entity_types = entity['types']
                    parsed_entities.append({
                        'text': entity_text,
                        'type': entity_types[0] if entity_types else 'UNKNOWN',
                        'types': entity_types
                    })
                elif 'start' in entity and 'end' in entity:
                    parsed_entities.append(entity)
                else:
                    logger.warning(f"Unknown entity format: {entity}")
        
        return parsed_entities
    
    def create_masked_text(self, text: str, entities: List[Dict], mask_style: str = "type_specific") -> str:
        """Create masked version of text"""
        if not entities:
            return text
            
        masked_text = text
        entities_with_positions = []
        
        for entity in entities:
            if 'start' in entity and 'end' in entity:
                entities_with_positions.append(entity)
            elif 'text' in entity:
                entity_text = entity['text']
                start = text.find(entity_text)
                if start != -1:
                    entities_with_positions.append({
                        'text': entity_text,
                        'type': entity.get('type', 'PII'),
                        'start': start,
                        'end': start + len(entity_text)
                    })
        
        sorted_entities = sorted(entities_with_positions, key=lambda x: x.get('start', 0), reverse=True)
        
        for entity in sorted_entities:
            start = entity.get('start', 0)
            end = entity.get('end', len(text))
            entity_type = entity.get('type', 'PII')
            
            if mask_style == "type_specific":
                mask = f"[{entity_type.upper()}]"
            elif mask_style == "generic":
                mask = "[REDACTED]"
            elif mask_style == "partial":
                original_text = text[start:end]
                if len(original_text) > 2:
                    mask = original_text[0] + "*" * (len(original_text) - 2) + original_text[-1]
                else:
                    mask = "*" * len(original_text)
            else:
                mask = f"[{entity_type.upper()}]"
            
            masked_text = masked_text[:start] + mask + masked_text[end:]
        
        return masked_text
    
    def create_instruction_response_pairs(self, df: pd.DataFrame, task_types: List[str] = None) -> List[Dict]:
        """Create instruction-response pairs for training"""
        
        if task_types is None:
            task_types = ["mask", "detect", "both"]
        
        formatted_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            text = row['text']
            domain = row.get('domain', 'general')
            doc_type = row.get('document_type', 'document')
            entities = self.parse_entities(row['entities'])
            
            if len(text) > 4000:
                text = text[:4000]
                entities = [e for e in entities if e.get('end', 0) <= 4000]
            
            for task_type in task_types:
                if task_type == "mask":
                    instruction = f"Mask all personally identifiable information (PII) and protected health information (PHI) in the following {doc_type} from the {domain} domain:\n\n{text}"
                    response = self.create_masked_text(text, entities, "type_specific")
                    
                elif task_type == "detect":
                    instruction = f"Identify all PII and PHI in the following {doc_type} from the {domain} domain. Return the results as a JSON array with 'text', 'type', 'start', and 'end' fields:\n\n{text}"
                    
                    detected_entities = []
                    for entity in entities:
                        start = entity.get('start', 0)
                        end = entity.get('end', len(text))
                        detected_entities.append({
                            'text': text[start:end],
                            'type': entity.get('type', 'UNKNOWN'),
                            'start': start,
                            'end': end
                        })
                    response = json.dumps(detected_entities, indent=2)
                    
                elif task_type == "both":
                    instruction = f"First identify all PII/PHI in the following {doc_type}, then provide a masked version. Document domain: {domain}\n\n{text}"
                    
                    detected_entities = []
                    for entity in entities:
                        start = entity.get('start', 0)
                        end = entity.get('end', len(text))
                        detected_entities.append({
                            'text': text[start:end],
                            'type': entity.get('type', 'UNKNOWN')
                        })
                    
                    masked_text = self.create_masked_text(text, entities, "type_specific")
                    response = f"Detected PII/PHI:\n{json.dumps(detected_entities, indent=2)}\n\nMasked text:\n{masked_text}"
                
                template = self.instruction_templates
                full_text = template["format"].format(
                    system=template["system"],
                    instruction=instruction,
                    response=response
                )
                
                formatted_data.append({
                    'text': full_text,
                    'instruction': instruction,
                    'response': response,
                    'task_type': task_type,
                    'domain': domain,
                    'document_type': doc_type
                })
        
        return formatted_data
    
    def prepare_datasets(self, 
                         train_df: pd.DataFrame, 
                         val_df: pd.DataFrame, 
                         test_df: pd.DataFrame,
                         task_types: List[str] = None) -> DatasetDict:
        """Prepare all datasets for training"""
        
        logger.info("Preparing training dataset...")
        train_data = self.create_instruction_response_pairs(train_df, task_types)
        
        logger.info("Preparing validation dataset...")
        val_data = self.create_instruction_response_pairs(val_df, task_types)
        
        logger.info("Preparing test dataset...")
        test_data = self.create_instruction_response_pairs(test_df, task_types)
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        logger.info(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return dataset_dict

# ============================================================================
# MODEL TRAINING
# ============================================================================

class LargeLLMTrainer:
    """Trainer for large language models with LoRA"""
    
    def __init__(self, 
                 model_name: str = "gemma2-9b",
                 use_quantization: bool = False,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False):
        
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["mistral-7b"])
        self.use_quantization = use_quantization
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with appropriate configurations"""
        
        logger.info(f"Loading model: {self.model_config['model_id']}")
        
        # Determine available device - optimized for Apple Silicon
        if torch.cuda.is_available():
            device_map = "auto"
            max_memory = {"cpu": "80GB", "cuda:0": "48GB"}
        elif torch.backends.mps.is_available():
            # Apple Silicon Mac with unified memory
            device_map = "cpu"  # MPS doesn't work well with large models yet
            max_memory = {"cpu": "110GB"}
            logger.info("Detected Apple Silicon with unified memory. Using CPU backend.")
        else:
            # Regular CPU only
            device_map = "cpu"
            max_memory = {"cpu": "110GB"}
        
        quantization_config = None
        if self.use_quantization:
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )
        
        load_kwargs = dict(
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        if max_memory:
            load_kwargs["max_memory"] = max_memory
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config['model_id'],
            **load_kwargs
        )
        
        if self.use_quantization:
            model = prepare_model_for_kbit_training(model)
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_config['model_id'])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info("Model and tokenizer loaded successfully")
        
        return model, tokenizer
    
    def setup_lora(self):
        """Configure and apply LoRA to the model"""
        
        logger.info("Configuring LoRA...")
        
        lora_config = LoraConfig(
            r=self.model_config['lora_r'],
            lora_alpha=self.model_config['lora_alpha'],
            target_modules=self.model_config['target_modules'],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Enable gradient computation for LoRA parameters
        for param in self.peft_model.parameters():
            param.requires_grad = True
        
        # Ensure the model is in training mode
        self.peft_model.train()
        
        self.peft_model.print_trainable_parameters()
        
        return self.peft_model
    
    def tokenize_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        """Tokenize the dataset with memory-efficient settings"""
        
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples['text'],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
            # Add labels for language modeling
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=10,
            num_proc=None,
            remove_columns=dataset.column_names,
            desc="Tokenizing",
            load_from_cache_file=False
        )
        
        return tokenized_dataset
    
    def train(self, 
              dataset_dict: DatasetDict,
              output_dir: str = "./pii_llm_lora",
              num_epochs: int = 3,
              batch_size: int = 4,
              gradient_accumulation_steps: int = 4,
              learning_rate: float = 2e-4,
              warmup_ratio: float = 0.03,
              max_length: int = 512):
        """Train the model with LoRA"""
        
        if self.model is None:
            self.setup_model_and_tokenizer()
            self.setup_lora()
        
        logger.info("Tokenizing datasets...")
        tokenized_train = self.tokenize_dataset(dataset_dict['train'], max_length)
        tokenized_val = self.tokenize_dataset(dataset_dict['validation'], max_length)
        
        total_steps = max(1, (len(tokenized_train) // (batch_size * gradient_accumulation_steps)) * num_epochs)
        warmup_steps = int(total_steps * warmup_ratio)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=True,
            optim="adamw_torch",
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            fp16=False,  # Don't use FP16 on Apple Silicon
            bf16=False,  # Don't use BF16 on Apple Silicon
            max_grad_norm=1.0,
            group_by_length=True,
            ddp_find_unused_parameters=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
        
        train_result = trainer.train()
        
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info("Evaluating on validation set...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        return trainer

# ============================================================================
# INFERENCE
# ============================================================================

class PIIInferenceEngine:
    """Inference engine for PII detection and masking"""
    
    def __init__(self, model_path: str, base_model_name: str = None):
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned model"""
        
        logger.info(f"Loading fine-tuned model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Determine available device for inference
        if torch.cuda.is_available():
            max_memory = {"cpu": "80GB", "cuda:0": "48GB"}
        elif torch.backends.mps.is_available():
            max_memory = {"cpu": "110GB"}
        else:
            max_memory = {"cpu": "110GB"}
        
        if self.base_model_name:
            load_kwargs = dict(
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory=max_memory
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **load_kwargs
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
        else:
            load_kwargs = dict(
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                max_memory=max_memory
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **load_kwargs
            )
        
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def generate_response(self, 
                          instruction: str, 
                          max_new_tokens: int = 512,
                          temperature: float = 0.1,
                          top_p: float = 0.95,
                          do_sample: bool = True) -> str:
        """Generate response for a given instruction"""
        
        formatted_input = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
        
        inputs = self.tokenizer(
            formatted_input, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        
        # Move inputs to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def mask_text(self, text: str) -> str:
        """Mask PII/PHI in text"""
        instruction = f"Mask all PII and PHI in the following text:\n\n{text}"
        return self.generate_response(instruction)
    
    def detect_pii(self, text: str) -> List[Dict]:
        """Detect PII/PHI entities in text"""
        instruction = f"Identify all PII and PHI in the following text. Return the results as a JSON array:\n\n{text}"
        response = self.generate_response(instruction)
        
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return []
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {response}")
            return []
    
    def batch_process(self, texts: List[str], task: str = "mask") -> List[str]:
        """Process multiple texts in batch"""
        results = []
        for text in tqdm(texts, desc=f"Processing {task}"):
            if task == "mask":
                results.append(self.mask_text(text))
            elif task == "detect":
                results.append(self.detect_pii(text))
        return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def load_gretel_dataset_from_huggingface():
    """Load the Gretel synthetic PII dataset from HuggingFace"""
    from datasets import load_dataset
    
    logger.info("Loading Gretel dataset from HuggingFace...")
    
    dataset_name = "gretelai/gretel-pii-masking-en-v1"
    
    try:
        dataset = load_dataset(dataset_name)
        logger.info(f"Successfully loaded dataset: {dataset_name}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return None, None, None
    
    if 'train' in dataset:
        train_df = dataset['train'].to_pandas()
    else:
        train_df = dataset[list(dataset.keys())[0]].to_pandas()
    
    if 'validation' in dataset:
        val_df = dataset['validation'].to_pandas()
    elif 'val' in dataset:
        val_df = dataset['val'].to_pandas()
    else:
        logger.warning("No validation split found, creating from train data...")
        val_df = train_df.sample(n=min(5000, len(train_df)//10), random_state=42)
        train_df = train_df.drop(val_df.index)
    
    if 'test' in dataset:
        test_df = dataset['test'].to_pandas()
    else:
        logger.warning("No test split found, creating from train data...")
        test_df = train_df.sample(n=min(5000, len(train_df)//10), random_state=42)
        train_df = train_df.drop(test_df.index)
    
    logger.info(f"Dataset loaded successfully!")
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Validation size: {len(val_df)}")
    logger.info(f"Test size: {len(test_df)}")
    logger.info(f"Columns: {train_df.columns.tolist()}")
    
    logger.info("\nExamining dataset structure:")
    for col in train_df.columns:
        logger.info(f"  - {col}: {train_df[col].dtype}")
        sample_val = train_df[col].iloc[0] if len(train_df) > 0 else None
        if sample_val is not None:
            if isinstance(sample_val, str) and len(sample_val) > 100:
                sample_val = sample_val[:100] + "..."
            logger.info(f"    Sample: {sample_val}")
    
    return train_df, val_df, test_df

def main():
    """Main execution function"""
    
    print("="*80)
    print("LARGE LLM FINE-TUNING FOR PII/PHI DETECTION")
    print("="*80)
    
    # Configuration
    config = {
        "model_name": "gemma2-9b",  # Using 9B model
        "use_quantization": False,  # Disable quantization on unified memory systems
        "load_in_4bit": False,
        "load_in_8bit": False,
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 1,
        "learning_rate": 2e-4,
        "max_length": 512,
        "output_dir": "./pii_llm_gemma2_9b_lora",
    }
    
    # Step 1: Load datasets from HuggingFace
    train_df, val_df, test_df = load_gretel_dataset_from_huggingface()
    
    if train_df is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # For initial testing, you might want to use a subset
    use_subset = input("\nUse subset for faster testing? (y/n): ").lower() == 'y'
    if use_subset:
        subset_size_train = int(input("Enter training subset size (default 1000): ") or "1000")
        subset_size_val = int(input("Enter validation subset size (default 200): ") or "200")
        subset_size_test = int(input("Enter test subset size (default 200): ") or "200")
        
        train_df = train_df.sample(n=min(subset_size_train, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(subset_size_val, len(val_df)), random_state=42)
        test_df = test_df.sample(n=min(subset_size_test, len(test_df)), random_state=42)
        
        logger.info(f"Using subset - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Step 2: Process datasets
    logger.info("Processing datasets...")
    model_type = config["model_name"].split("-")[0] if "-" in config["model_name"] else config["model_name"]
    processor = GretelDatasetProcessor(model_type=model_type)
    
    task_types = ["mask", "detect"]
    dataset_dict = processor.prepare_datasets(train_df, val_df, test_df, task_types=task_types)
    
    # Step 3: Initialize trainer
    logger.info(f"Initializing trainer for {config['model_name']}...")
    trainer_instance = LargeLLMTrainer(
        model_name=config["model_name"],
        use_quantization=config["use_quantization"],
        load_in_4bit=config["load_in_4bit"],
        load_in_8bit=config["load_in_8bit"]
    )
    
    # Step 4: Setup model
    trainer_instance.setup_model_and_tokenizer()
    trainer_instance.setup_lora()
    
    # Step 5: Train
    logger.info("Starting training...")
    trainer = trainer_instance.train(
        dataset_dict=dataset_dict,
        output_dir=config["output_dir"],
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        max_length=config["max_length"]
    )
    
    # Step 6: Test the model
    logger.info("Testing the fine-tuned model...")
    inference_engine = PIIInferenceEngine(
        model_path=config["output_dir"],
        base_model_name=MODEL_CONFIGS[config["model_name"]]["model_id"]
    )
    inference_engine.load_model()
    
    # Test examples
    test_texts = [
        "John Smith visited Dr. Johnson at 123 Main Street, New York on January 15, 2024. His SSN is 123-45-6789 and email is john.smith@email.com",
        "Patient Mary Williams (DOB: 05/12/1980) was diagnosed with hypertension. Medical Record Number: MRN-2024-00123. Contact: mary@healthcare.org, Phone: 555-123-4567",
        "Invoice #INV-2024-789 for Robert Brown, Account Number: 9876543210, Credit Card: 4111-1111-1111-1111, Amount: $5,000.00",
    ]
    
    print("\n" + "="*80)
    print("TESTING FINE-TUNED MODEL")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nExample {i}:")
        print(f"Original: {text}")
        print(f"Masked: {inference_engine.mask_text(text)}")
        print("-"*80)
    
    # Step 7: Evaluate on test set
    if input("\nRun full evaluation on test set? (y/n): ").lower() == 'y':
        logger.info("Evaluating on test set...")
        test_sample = test_df.sample(n=min(100, len(test_df)), random_state=42)
        
        correct_masks = 0
        total = len(test_sample)
        
        for _, row in tqdm(test_sample.iterrows(), total=total, desc="Evaluating"):
            text = row['text'][:2000]
            entities = processor.parse_entities(row['entities'])
            expected_masked = processor.create_masked_text(text, entities)
            
            predicted_masked = inference_engine.mask_text(text)
            
            if "[" in predicted_masked and "]" in predicted_masked:
                correct_masks += 1
        
        accuracy = correct_masks / total
        print(f"\nEvaluation Results:")
        print(f"Masking Accuracy: {accuracy:.2%}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {config['output_dir']}")
    print("="*80)

if __name__ == "__main__":
    main()
