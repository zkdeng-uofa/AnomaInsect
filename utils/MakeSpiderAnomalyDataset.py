import os
import argparse
import random
import shutil
import json

random.seed(42)

def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['root_dir', 'abnormal_name', 'output_dir', 'normal_sample_size', 'abnormal_sample_size']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in config file")
        
        # Validate data types
        if not isinstance(config['normal_sample_size'], int) or config['normal_sample_size'] <= 0:
            raise ValueError("normal_sample_size must be a positive integer")
        if not isinstance(config['abnormal_sample_size'], int) or config['abnormal_sample_size'] <= 0:
            raise ValueError("abnormal_sample_size must be a positive integer")
        
        # Optional fields with defaults
        config.setdefault('test_sample_ratio', 0.2)
        config.setdefault('random_seed', 42)
        config.setdefault('file_extensions', ['.jpg', '.png', '.jpeg'])
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Create anomaly detection dataset from images")
    
    # Add config file option
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Original CLI arguments (optional when using config)
    parser.add_argument("--root_dir", type=str, help="Root directory containing images")
    parser.add_argument("--abnormal_name", type=str, help="Substring to identify abnormal image directories")
    parser.add_argument("--output_dir", type=str, help="Output directory for organized dataset")
    parser.add_argument("--normal_sample_size", type=int, help="Number of normal images to sample")
    parser.add_argument("--abnormal_sample_size", type=int, help="Number of abnormal images to sample")
    parser.add_argument("--test_sample_size", type=float, default=100, help="Ratio of test samples (default: 0.2)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override config with any provided CLI arguments
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
        # Convert back to namespace for compatibility
        for key, value in config.items():
            setattr(args, key, value)
    else:
        # Ensure required arguments are provided when not using config
        required_args = ['root_dir', 'abnormal_name', 'output_dir', 'normal_sample_size', 'abnormal_sample_size']
        for arg in required_args:
            if getattr(args, arg) is None:
                parser.error(f"--{arg} is required when not using --config")
    
    return args

def ListFiles(root_dir, abnormal_name, file_extensions=None):
    if file_extensions is None:
        file_extensions = ['.jpg', '.png', '.jpeg']
    
    normal_files = []
    abnormal_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext.lower()) for ext in file_extensions):
                full_path = os.path.join(root, file)
                if abnormal_name in root:
                    abnormal_files.append(full_path)
                else:
                    normal_files.append(full_path)

    return normal_files, abnormal_files

def sample_and_copy_files(normal_files, abnormal_files, output_dir, normal_sample_size, abnormal_sample_size, test_sample_ratio=0.2, test_sample_size=100):
    # Calculate test sample size based on the specified ratio
    all_files = normal_files + abnormal_files
    test_sample = random.sample(all_files, min(test_sample_size, len(all_files)))
    
    # Remove test files from the available files for normal/abnormal sampling
    remaining_normal_files = [f for f in normal_files if f not in test_sample]
    remaining_abnormal_files = [f for f in abnormal_files if f not in test_sample]
    
    # Sample from remaining files for normal and abnormal folders
    normal_sample = random.sample(remaining_normal_files, min(normal_sample_size, len(remaining_normal_files)))
    abnormal_sample = random.sample(remaining_abnormal_files, min(abnormal_sample_size, len(remaining_abnormal_files)))

    # Copy files to respective folders
    for file_path in normal_sample:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(output_dir, "normal", filename))

    for file_path in abnormal_sample:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(output_dir, "abnormal", filename))

    for file_path in test_sample:
        filename = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(output_dir, "test", filename))

    return None

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(getattr(args, 'random_seed', 42))
    
    # Get file extensions from config or use default
    file_extensions = getattr(args, 'file_extensions', ['.jpg', '.png', '.jpeg'])
    test_sample_ratio = getattr(args, 'test_sample_ratio', 0.2)
    
    print(f"Loading files from: {args.root_dir}")
    print(f"Abnormal identifier: {args.abnormal_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample sizes - Normal: {args.normal_sample_size}, Abnormal: {args.abnormal_sample_size}")
    print(f"Test sample ratio: {test_sample_ratio}")
    
    normal_files, abnormal_files = ListFiles(args.root_dir, args.abnormal_name, file_extensions)
    
    print(f"Found {len(normal_files)} normal files and {len(abnormal_files)} abnormal files")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "abnormal"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "test"), exist_ok=True)

    sample_and_copy_files(normal_files, abnormal_files, args.output_dir, 
                         args.normal_sample_size, args.abnormal_sample_size, test_sample_ratio)
    
    print("Dataset creation completed successfully!")
    return None

if __name__ == "__main__":
    main()
