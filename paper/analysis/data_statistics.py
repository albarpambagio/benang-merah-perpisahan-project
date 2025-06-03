#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import altair as alt
from typing import Dict, List, Any, Optional
import logging
import argparse
from dataclasses import dataclass
import re
from concurrent.futures import ProcessPoolExecutor
import tqdm

# %%
@dataclass
class PlotConfig:
    """Configuration for plot styling and dimensions."""
    width: int = 600
    height: int = 400
    color_scheme: str = 'category10'
    interactive: bool = True

@dataclass
class DataConfig:
    """Configuration for data processing."""
    required_fields: List[str] = None
    chunk_size: int = 1000  # For parallel processing
    max_workers: int = 4    # For parallel processing

    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = ['text', 'section', 'verdict']

# %%
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_record(record: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate if a record contains all required fields.
    
    Args:
        record (Dict[str, Any]): Data record to validate
        required_fields (List[str]): List of required field names
        
    Returns:
        bool: True if record is valid, False otherwise
    """
    return all(field in record and record[field] for field in required_fields)

def process_chunk(chunk: List[Dict[str, Any]], required_fields: List[str]) -> List[Dict[str, Any]]:
    """
    Process a chunk of records in parallel.
    
    Args:
        chunk (List[Dict[str, Any]]): List of records to process
        required_fields (List[str]): List of required field names
        
    Returns:
        List[Dict[str, Any]]: List of valid records
    """
    return [record for record in chunk if validate_record(record, required_fields)]

def load_jsonl_data(file_path: str, config: DataConfig) -> List[Dict[str, Any]]:
    """
    Load and validate data from a JSONL file with parallel processing.
    
    Args:
        file_path (str): Path to the JSONL file
        config (DataConfig): Data processing configuration
        
    Returns:
        List[Dict[str, Any]]: List of valid dictionaries containing the data
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read all lines first to get total count
            lines = f.readlines()
            total_lines = len(lines)
            
            # Process in chunks
            with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
                chunks = [lines[i:i + config.chunk_size] 
                         for i in range(0, total_lines, config.chunk_size)]
                
                # Process chunks in parallel with progress bar
                futures = []
                for chunk in chunks:
                    chunk_data = [json.loads(line) for line in chunk]
                    futures.append(executor.submit(process_chunk, chunk_data, config.required_fields))
                
                # Collect results
                for future in tqdm.tqdm(futures, total=len(chunks), desc="Loading data"):
                    data.extend(future.result())
        
        logger.info(f"Successfully loaded {len(data)} valid records from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        return []

def calculate_basic_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics about the dataset.
    
    Args:
        data (List[Dict[str, Any]]): List of data records
        
    Returns:
        Dict[str, Any]: Dictionary containing basic statistics
    """
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Text statistics
    text_lengths = df['text'].str.len()
    word_counts = df['text'].str.split().str.len()
    unique_words = df['text'].str.split().apply(lambda x: len(set(x)))
    
    stats = {
        'total_records': len(data),
        'text_length_stats': {
            'mean': float(text_lengths.mean()),
            'median': float(text_lengths.median()),
            'min': int(text_lengths.min()),
            'max': int(text_lengths.max()),
            'std': float(text_lengths.std())
        },
        'word_stats': {
            'mean_words': float(word_counts.mean()),
            'median_words': float(word_counts.median()),
            'min_words': int(word_counts.min()),
            'max_words': int(word_counts.max()),
            'mean_unique_words': float(unique_words.mean())
        },
        'section_counts': dict(Counter(df['section'])),
        'verdict_counts': dict(Counter(df['verdict']))
    }
    return stats

def plot_text_length_distribution(data: List[Dict[str, Any]], save_path: Optional[str] = None, 
                                config: PlotConfig = PlotConfig()):
    """
    Plot the distribution of text lengths using Altair.
    
    Args:
        data (List[Dict[str, Any]]): List of data records
        save_path (Optional[str]): Path to save the plot
        config (PlotConfig): Plot configuration
    """
    df = pd.DataFrame({
        'text_length': [len(d.get('text', '')) for d in data],
        'word_count': [len(d.get('text', '').split()) for d in data]
    })
    
    # Save CSV
    if save_path:
        csv_path = save_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False)
    
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('text_length:Q', bin=alt.Bin(maxbins=50), title='Text Length'),
        alt.Y('count()', title='Count'),
        color=alt.Color('text_length:Q', scale=alt.Scale(scheme=config.color_scheme))
    ).properties(
        title='Distribution of Text Lengths',
        width=config.width,
        height=config.height
    ).interactive() if config.interactive else chart
    
    if save_path:
        chart.save(save_path)
    return chart

def plot_section_distribution(data: List[Dict[str, Any]], save_path: Optional[str] = None,
                            config: PlotConfig = PlotConfig()):
    """
    Plot the distribution of sections using Altair.
    
    Args:
        data (List[Dict[str, Any]]): List of data records
        save_path (Optional[str]): Path to save the plot
        config (PlotConfig): Plot configuration
    """
    section_counts = Counter([d.get('section', '') for d in data])
    df = pd.DataFrame({
        'section': list(section_counts.keys()),
        'count': list(section_counts.values())
    })
    
    # Save CSV
    if save_path:
        csv_path = save_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False)
    
    chart = alt.Chart(df).mark_bar().encode(
        alt.X('section:N', title='Section', sort='-y'),
        alt.Y('count:Q', title='Count'),
        color=alt.Color('section:N', scale=alt.Scale(scheme=config.color_scheme))
    ).properties(
        title='Distribution of Sections',
        width=config.width,
        height=config.height
    ).interactive() if config.interactive else chart
    
    if save_path:
        chart.save(save_path)
    return chart

def plot_verdict_distribution(data: List[Dict[str, Any]], save_path: Optional[str] = None,
                            config: PlotConfig = PlotConfig()):
    """
    Plot the distribution of verdicts using Altair.
    
    Args:
        data (List[Dict[str, Any]]): List of data records
        save_path (Optional[str]): Path to save the plot
        config (PlotConfig): Plot configuration
    """
    verdict_counts = Counter([d.get('verdict', '') for d in data])
    df = pd.DataFrame({
        'verdict': list(verdict_counts.keys()),
        'count': list(verdict_counts.values())
    })
    
    # Save CSV
    if save_path:
        csv_path = save_path.replace('.png', '.csv')
        df.to_csv(csv_path, index=False)
    
    chart = alt.Chart(df).mark_arc().encode(
        theta=alt.Theta(field="count", type="quantitative"),
        color=alt.Color(field="verdict", type="nominal", scale=alt.Scale(scheme=config.color_scheme)),
        tooltip=['verdict', 'count']
    ).properties(
        title='Distribution of Verdicts',
        width=config.width,
        height=config.height
    ).interactive() if config.interactive else chart
    
    if save_path:
        chart.save(save_path)
    return chart

def generate_statistics_report(data: List[Dict[str, Any]], output_dir: str,
                             plot_config: PlotConfig = PlotConfig()):
    """
    Generate a comprehensive statistics report with both CSV and PNG outputs.
    
    Args:
        data (List[Dict[str, Any]]): List of data records
        output_dir (str): Directory to save the report and plots
        plot_config (PlotConfig): Configuration for plots
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate basic statistics
    stats = calculate_basic_statistics(data)
    
    # Save statistics to JSON
    with open(output_path / "statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)
    
    # Convert statistics to DataFrame and save as CSV
    stats_df = pd.DataFrame({
        'metric': ['total_records'] + 
                 [f'text_length_{k}' for k in stats['text_length_stats'].keys()] +
                 [f'word_{k}' for k in stats['word_stats'].keys()],
        'value': [stats['total_records']] + 
                list(stats['text_length_stats'].values()) +
                list(stats['word_stats'].values())
    })
    stats_df.to_csv(output_path / "statistics.csv", index=False)
    
    # Save section counts as CSV
    section_df = pd.DataFrame({
        'section': list(stats['section_counts'].keys()),
        'count': list(stats['section_counts'].values())
    })
    section_df.to_csv(output_path / "section_counts.csv", index=False)
    
    # Save verdict counts as CSV
    verdict_df = pd.DataFrame({
        'verdict': list(stats['verdict_counts'].keys()),
        'count': list(stats['verdict_counts'].values())
    })
    verdict_df.to_csv(output_path / "verdict_counts.csv", index=False)
    
    # Generate plots (will also save their respective CSVs)
    plot_text_length_distribution(data, str(output_path / "text_length_distribution.png"), plot_config)
    plot_section_distribution(data, str(output_path / "section_distribution.png"), plot_config)
    plot_verdict_distribution(data, str(output_path / "verdict_distribution.png"), plot_config)
    
    logger.info(f"Statistics report generated in {output_dir}")
    logger.info("Generated files:")
    logger.info("- statistics.json: Complete statistics in JSON format")
    logger.info("- statistics.csv: Basic statistics in CSV format")
    logger.info("- section_counts.csv: Section distribution data")
    logger.info("- verdict_counts.csv: Verdict distribution data")
    logger.info("- text_length_distribution.png/csv: Text length visualization and data")
    logger.info("- section_distribution.png/csv: Section distribution visualization and data")
    logger.info("- verdict_distribution.png/csv: Verdict distribution visualization and data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data statistics report')
    parser.add_argument('input_file', help='Path to input JSONL file')
    parser.add_argument('output_dir', help='Directory to save report')
    parser.add_argument('--interactive', action='store_true', help='Show interactive plots')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for parallel processing')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of worker processes')
    args = parser.parse_args()
    
    # Configure data processing
    data_config = DataConfig(chunk_size=args.chunk_size, max_workers=args.max_workers)
    
    # Configure plotting
    plot_config = PlotConfig(interactive=args.interactive)
    
    # Load and process data
    data = load_jsonl_data(args.input_file, data_config)
    if data:  # Only proceed if data loaded successfully
        generate_statistics_report(data, args.output_dir, plot_config) 