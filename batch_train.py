# ---------------------------------------------------------------------------------------
# AutoRL: Batch trainging version
# Author: Rajesh Siraskar
# ---------------------------------------------------------------------------------------
# V.1.0: 11-Mar-2026:
# ---------------------------------------------------------------------------------------

print('\n--------------------------------------------------------------------------')
print('AutoRL - Batch Training version')
print('--------------------------------------------------------------------------\n')

print(' - Loading libraries ...')
import os

from matplotlib.colors import LinearSegmentedColormap
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import argparse
import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import List, Dict, Tuple, Optional
import glob
import sqlite3
try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
except ImportError:
    pass  # Will handle gracefully in plot creation function

# Import RL module
import rl_pdm

print('- Loaded.\n - Starting AutoRL CLI Pipeline ...\n')

# ======================================================================================
# GLOBAL Constants (Checkpoint Database, Evaluation Rounds, Retry Limits)
# ======================================================================================
# timestamp_fmt = "%Y%m%d_%H%M%S"
timestamp_fmt = "%b-%d_%H%M%S"
BATCH_TRAIN = 3 # Number of training models to train before running analysis and picking next batch (if -B N specified)
EVAL_ROUNDS = 20  # Default 20 - Number of evaluation rounds for multi-round evaluation (if -V N specified)
INDIVDUAL_PLOTS = False
PDF_REPORTS = True

# Checkpoint Model Evaluation - recovery from failuer - Maximum retries
RETRY_EVAL = 10 # default 10
CHECKPOINT_DB = "checkpoint.db"

def init_checkpoint_db():
    """
    Initialize the checkpoint database and create tables if they don't exist.
    
    Returns:
        Connection object to the database
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    # Create batches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            schema TEXT NOT NULL,
            algorithms TEXT NOT NULL,
            episodes INTEGER NOT NULL,
            learning_rates TEXT NOT NULL,
            gammas TEXT NOT NULL,
            attention_mech INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            task_id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL,
            task_type TEXT NOT NULL,
            task_order INTEGER NOT NULL,
            algo TEXT,
            training_file TEXT,
            attention_type TEXT,
            learning_rate REAL,
            gamma REAL,
            model_path TEXT,
            status TEXT NOT NULL,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
        )
    ''')
    
    conn.commit()
    return conn


def create_batch_id():
    """
    Generate a unique batch ID using timestamp.
    
    Returns:
        String batch ID in format: BATCH_YYYYMMDD_HHMMSS
    """
    now = datetime.now()
    return f"BATCH_{now.strftime('%Y%m%d_%H%M%S')}"


def create_batch_record(batch_id: str, schema: str, algos: List[str], 
                       episodes: int, lrs: List[float], gammas: List[float], 
                       attention_mech: int):
    """
    Create a new batch record in the database.
    
    Args:
        batch_id: Unique batch identifier
        schema: 'SIT' or 'IEEE'
        algos: List of algorithm names
        episodes: Number of training episodes
        lrs: List of learning rates
        gammas: List of gamma values
        attention_mech: 1 for attention, 0 for none
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO batches (batch_id, schema, algorithms, episodes, learning_rates, 
                           gammas, attention_mech, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        batch_id,
        schema,
        ','.join(algos),
        episodes,
        ','.join(map(str, lrs)),
        ','.join(map(str, gammas)),
        attention_mech,
        'Queued'
    ))
    
    conn.commit()
    conn.close()


def add_task_to_queue(batch_id: str, task_type: str, task_order: int, 
                     algo: str = None, training_file: str = None, 
                     attention_type: str = None, learning_rate: float = None, 
                     gamma: float = None):
    """
    Add a task to the queue in the database.
    
    Args:
        batch_id: Batch identifier
        task_type: 'TRAIN', 'EVAL', 'HEATMAP', 'REPORT'
        task_order: Execution order
        algo: Algorithm name (for TRAIN tasks)
        training_file: Training data filename
        attention_type: Attention mechanism type
        learning_rate: Learning rate value
        gamma: Gamma value
    
    Returns:
        task_id of the created task
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO tasks (batch_id, task_type, task_order, algo, training_file, 
                         attention_type, learning_rate, gamma, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        batch_id, task_type, task_order, algo, training_file, 
        attention_type, learning_rate, gamma, 'Queued'
    ))
    
    task_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return task_id


def update_task_status(task_id: int, status: str, model_path: str = None, 
                      error_msg: str = None):
    """
    Update the status of a task.
    
    Args:
        task_id: Task identifier
        status: New status ('Queued', 'Training', 'Done', 'Failed')
        model_path: Path to saved model (optional)
        error_msg: Error message if failed (optional)
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE tasks 
        SET status = ?, model_path = ?, error_message = ?, updated_at = CURRENT_TIMESTAMP
        WHERE task_id = ?
    ''', (status, model_path, error_msg, task_id))
    
    conn.commit()
    conn.close()


def update_batch_status(batch_id: str, status: str):
    """
    Update the status of a batch.
    
    Args:
        batch_id: Batch identifier
        status: New status ('Queued', 'WIP', 'Done', 'Failed')
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE batches 
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE batch_id = ?
    ''', (status, batch_id))
    
    conn.commit()
    conn.close()


def get_incomplete_batch() -> Optional[Dict]:
    """
    Get the most recent incomplete batch (status = 'WIP' or 'Queued').
    
    Returns:
        Dictionary with batch information or None if no incomplete batch found
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM batches 
        WHERE status IN ('WIP', 'Queued')
        ORDER BY created_at DESC
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None


def get_pending_tasks(batch_id: str) -> List[Dict]:
    """
    Get all pending tasks for a batch (status != 'Done').
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        List of task dictionaries
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM tasks 
        WHERE batch_id = ? AND status != 'Done'
        ORDER BY task_order
    ''', (batch_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def get_completed_tasks(batch_id: str, task_type: str = None) -> List[Dict]:
    """
    Get all completed tasks for a batch.
    
    Args:
        batch_id: Batch identifier
        task_type: Optional filter by task type
    
    Returns:
        List of completed task dictionaries
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if task_type:
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE batch_id = ? AND status = 'Done' AND task_type = ?
            ORDER BY task_order
        ''', (batch_id, task_type))
    else:
        cursor.execute('''
            SELECT * FROM tasks 
            WHERE batch_id = ? AND status = 'Done'
            ORDER BY task_order
        ''', (batch_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]


def mark_batch_complete(batch_id: str):
    """
    Mark a batch as complete if all tasks are done.
    
    Args:
        batch_id: Batch identifier
    
    Returns:
        True if batch was marked complete, False otherwise
    """
    conn = sqlite3.connect(CHECKPOINT_DB)
    cursor = conn.cursor()
    
    # Check if all tasks are done
    cursor.execute('''
        SELECT COUNT(*) as total, 
               SUM(CASE WHEN status = 'Done' THEN 1 ELSE 0 END) as done
        FROM tasks
        WHERE batch_id = ?
    ''', (batch_id,))
    
    result = cursor.fetchone()
    total, done = result
    
    if total > 0 and total == done:
        update_batch_status(batch_id, 'Done')
        conn.close()
        return True
    
    conn.close()
    return False



def get_schema_files(schema: str, data_dir: str = "data") -> List[str]:
    """
    Get all CSV files for the specified schema (SIT or IEEE).
    
    Args:
        schema: 'SIT' or 'IEEE'
        data_dir: Path to data directory
    
    Returns:
        List of full file paths for the schema
    """
    # Check if subdirectory exists for strict organizational compliance
    subdir = os.path.join(data_dir, schema)
    if os.path.isdir(subdir):
        files = glob.glob(os.path.join(subdir, "*.csv"))
    else:
        # Fallback to flat directory
        schema_pattern = os.path.join(data_dir, f"{schema}_*.csv")
        files = glob.glob(schema_pattern)
        
    # Exclude test files and tiny files
    files = [f for f in files if not f.endswith(('_TEST.csv', '_tiny.csv', 'temp_sensor_data.csv'))]
    files.sort()
    return files


def train_agents(schema: str, algos: List[str], episodes: int, attention_mech: int, 
                 learning_rates: List[float], gammas: List[float], batch_id: str = None, 
                 resume_mode: bool = False, attention_types: List[str] = None) -> Dict:
    """
    Train agents on all data files of the specified schema with grid search over hyperparameters.
    
    Args:
        schema: 'SIT' or 'IEEE'
        algos: List of algorithm names
        episodes: Number of training episodes
        attention_mech: 1 to use attention mechanism, 0 for none
        learning_rates: List of learning rate values to try
        gammas: List of gamma values to try
        batch_id: Batch ID (required for both new and resumed batches)
        resume_mode: True if resuming an incomplete batch, False for new batch
    
    Returns:
        Dictionary mapping (algo, training_file, attention_type, lr, gamma) -> model_path
    """
    print(f"\n{'='*80}")
    print(f"TRAINING PHASE: Schema={schema}, Algos={algos}, Episodes={episodes}, Attention={attention_mech}")
    print(f"Grid Search: LRs={learning_rates}, Gammas={gammas}")
    if batch_id:
        mode_label = "RECOVERY MODE" if resume_mode else "NEW BATCH"
        print(f"Batch ID: {batch_id} ({mode_label})")
    print(f"{'='*80}\n")
    
    # Set global episodes in rl_pdm
    rl_pdm.EPISODES = episodes
    

    # Get training files
    training_files = get_schema_files(schema)
    if not training_files:
        print(f"ERROR: No training files found for schema {schema}")
        sys.exit(1)

    print(f"Training files for {schema}: {[Path(f).stem for f in training_files]}\n")

    trained_models = {}

    # Mapping from short forms (used in CLI) to rl_pdm attention type names
    attention_short_to_full = {
        'NW': 'NW',           # NadarayaWatson
        'TP': 'Temporal',     # Temporal
        'MH': 'MultiHead',    # MultiHead
        'SA': 'SelfAttn'      # SelfAttn
    }

    # Use caller-supplied attention_types if provided; otherwise derive from attention_mech flag
    if attention_types is None:
        if attention_mech == 1:
            attention_types = ['NW', 'TP', 'MH', 'SA']
        else:
            attention_types = [None]

    # Batch training mode: process BATCH_TRAIN files at a time
    total_files = len(training_files)
    batch_size = BATCH_TRAIN
    file_batches = [training_files[i:i+batch_size] for i in range(0, total_files, batch_size)]

    for batch_idx, batch_files in enumerate(file_batches, 1):
        print(f"\n{'='*80}")
        print(f"BATCH {batch_idx}: Training on files: {[Path(f).stem for f in batch_files]}")
        print(f"{'='*80}\n")

        batch_trained_models = {}
        training_queue = []
        task_order = 0

        for training_file in batch_files:
            training_filename = Path(training_file).stem
            for algo in algos:
                for att_short in attention_types:
                    for lr in learning_rates:
                        for gm in gammas:
                            task_order += 1
                            att_label = f" ({att_short})" if att_short else ""

                            # Add task to database
                            task_id = add_task_to_queue(
                                batch_id=batch_id,
                                task_type='TRAIN',
                                task_order=task_order,
                                algo=algo,
                                training_file=training_filename,
                                attention_type=att_short,
                                learning_rate=lr,
                                gamma=gm
                            )

                            queue_item = {
                                'task_id': task_id,
                                'training_file': training_file,
                                'training_filename': training_filename,
                                'algo': algo,
                                'att_short': att_short,
                                'lr': lr,
                                'gm': gm,
                                'display': f"[{task_order:03d}] {algo}{att_label} | File: {training_filename} | LR={lr} | Gamma={gm}"
                            }
                            training_queue.append(queue_item)
                            print(queue_item['display'])

        print(f"\n{'='*80}")
        print(f"TOTAL TRAINING JOBS IN BATCH: {len(training_queue)}")
        print(f"{'='*80}\n")

        # Train combinations - iterate through queue
        for idx, queue_item in enumerate(training_queue, 1):
            training_file = queue_item['training_file']
            training_filename = queue_item['training_filename']
            algo = queue_item['algo']
            att_short = queue_item['att_short']
            lr = queue_item['lr']
            gm = queue_item['gm']
            task_id = queue_item['task_id']

            # Map short form to full rl_pdm name
            att_full = attention_short_to_full.get(att_short) if att_short else None

            # Format attention label for display
            att_label = f" ({att_short})" if att_short else ""
            print(f"\n>>> [{idx:03d}/{len(training_queue)}] Training {algo} on {training_filename}{att_label} | LR={lr} | Gamma={gm}")

            # Update task status to 'Training'
            update_task_status(task_id, 'Training')

            try:
                # Train with specific hyperparameters
                result = rl_pdm.train_single_model(
                    data_file=training_file,
                    algo_name=algo,
                    lr=lr,
                    gm=gm,
                    callback_func=None,  # No callback for CLI
                    attention_type=att_full,  # Pass full name to rl_pdm
                    data_filename=training_filename
                )

                if 'error' in result:
                    print(f"  [x] Training failed: {result['error']}")
                    update_task_status(task_id, 'Failed', error_msg=result['error'])
                else:
                    model_path = result['model_path']
                    batch_trained_models[(algo, training_filename, att_short, lr, gm)] = model_path
                    trained_models[(algo, training_filename, att_short, lr, gm)] = model_path
                    print(f"  [+] Model saved: {model_path}")
                    print(f"    Weighted Score: {result['Weighted Score']:.4f}")

                    # Update task status to 'Done' with model path
                    update_task_status(task_id, 'Done', model_path=model_path)

            except Exception as e:
                error_msg = str(e)
                print(f"  [x] Training error: {error_msg}")
                update_task_status(task_id, 'Failed', error_msg=error_msg)

        print(f"\n{'='*80}")
        print(f"Batch {batch_idx} training complete. Models trained: {len(batch_trained_models)}")
        print(f"{'='*80}\n")

        # Run analysis and save main analysis files (skip individual plots)
        if batch_trained_models:
            results_df = evaluate_agents(schema, batch_trained_models, skip_individual_plots=True, num_eval_rounds=EVAL_ROUNDS)
            save_results(results_df, schema, attention_mech, multiround=(EVAL_ROUNDS > 1), batch_idx=batch_idx)
            create_heatmaps(results_df, schema, attention_mech, batch_idx=batch_idx, batch_files=batch_files)
            generate_analysis_report(results_df, schema, attention_mech, batch_idx=batch_idx, batch_files=batch_files)

    print(f"\n{'='*80}")
    print(f"All batches complete. Total models trained: {len(trained_models)}")
    print(f"{'='*80}\n")

    return trained_models


def discover_trained_models(schema: str, models_dir: str = "models") -> Dict:
    """
    Discover already trained models for the specified schema.
    
    Args:
        schema: 'SIT' or 'IEEE'
        models_dir: Directory containing models
    
    Returns:
        Dictionary mapping (algo, training_file, attention_type) -> model_path
    """
    print(f"\n{'='*80}")
    print(f"DISCOVERY PHASE: Looking for existing {schema} models in '{models_dir}'")
    print(f"{'='*80}\n")
    
    if not os.path.exists(models_dir):
        print(f"ERROR: Models directory '{models_dir}' not found.")
        return {}

    trained_models = {}
    
    # Mapping from full attention names (in metadata) to short forms (for display/keys)
    attention_full_to_short = {
        'NW': 'NW', 'NadarayaWatson': 'NW',
        'DL': 'DL', 'Simple': 'DL',
        'TP': 'TP', 'Temporal': 'TP', 
        'MH': 'MH', 'MultiHead': 'MH',
        'SA': 'SA', 'SelfAttn': 'SA',
        'HY': 'HY', 'Hybrid': 'HY'
    }

    count = 0
    # Walk through models dir
    try:
        for filename in os.listdir(models_dir):
            if filename.endswith(".zip"):
                try:
                    model_path = os.path.join(models_dir, filename)
                    model_base = os.path.splitext(model_path)[0] # remove .zip
                    metadata_path = model_base + "_metadata.json"
                    
                    algo = "Unknown"
                    training_file = "Unknown"
                    att_short = None
                    
                    # 1. Try Metadata first (Robust)
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r') as f:
                                meta = json.load(f)
                                
                            algo = meta.get('algorithm', 'Unknown')
                            training_file = meta.get('training_data', 'Unknown')
                            att_full = meta.get('attention_mechanism')
                            
                            if att_full:
                                att_short = attention_full_to_short.get(att_full, att_full)
                            else:
                                att_short = None
                                
                            # Filter by schema in training_file
                            if schema not in training_file:
                                continue

                        except Exception as e:
                            print(f"  [!] Error reading metadata for {filename}: {e}")
                            continue
                    
                    else:
                        # 2. Fallback: Parse filename
                        # Format: Algo_TrainingData_... (e.g. PPO_IEEE_tiny_...)
                        if schema not in filename:
                            continue
                            
                        parts = filename.split('_')
                        if len(parts) < 3:
                             continue
                        
                        algo = parts[0]
                        # If schema is IEEE, training file starts with IEEE
                        training_file = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else f"{schema}_Unknown"
                        
                        # Check for attention suffix in filename
                        if "_NW" in filename: att_short = "NW"
                        elif "_DL" in filename: att_short = "DL"
                        elif "_TP" in filename: att_short = "TP"
                        elif "_MH" in filename: att_short = "MH"
                        elif "_SA" in filename: att_short = "SA"
                        elif "_HY" in filename: att_short = "HY"
                    
                    # Add to dict
                    key = (algo, training_file, att_short)
                    trained_models[key] = model_path
                    count += 1
                    
                    att_label = f" ({att_short})" if att_short else ""
                    print(f"  Found: {algo} - {training_file}{att_label}")

                except Exception as e:
                     print(f"  [!] Skipped {filename}: {e}")
                     continue

    except Exception as e:
        print(f"ERROR: Failed to list models directory: {e}")
        return {}

    print(f"\nFound {count} models matching schema '{schema}'.")
    return trained_models


def create_evaluation_plot(eval_result: Dict, model_filename: str, test_filename: str, results_dir: str = "results"):
    """
    Create and save an evaluation plot showing tool wear and replacements.
    
    Args:
        eval_result: Dictionary from adjusted_evaluate_model
        model_filename: Filename of the model
        test_filename: Filename of the test data
        results_dir: Directory to save plots
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        os.makedirs(results_dir, exist_ok=True)
        
        timesteps = eval_result['timesteps']
        tool_wear = eval_result['tool_wear']
        actions = eval_result['actions']
        wear_threshold = eval_result['wear_threshold']
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Add tool wear as blue line on primary y-axis
        fig.add_trace(
            go.Scatter(
                x=timesteps,
                y=tool_wear,
                name="Tool Wear",
                line=dict(color='#636EFA', width=3),
                mode='lines'
            ),
            secondary_y=False
        )
        
        # Add wear threshold as dotted line on primary y-axis
        fig.add_hline(
            y=wear_threshold,
            line_dash="dot",
            line_color="gray",
            annotation_text="Threshold",
            annotation_position="right",
            secondary_y=False
        )
        
        # Add IAR bounds if present
        IAR_lower = eval_result.get('IAR_lower', None)
        IAR_upper = eval_result.get('IAR_upper', None)
        if IAR_lower is not None and IAR_upper is not None:
            fig.add_hline(
                y=IAR_lower,
                line_dash="dot",
                line_color="teal",
                opacity=0.5,
                annotation_text="IAR Lower",
                annotation_position="right",
                secondary_y=False
            )
            fig.add_hline(
                y=IAR_upper,
                line_dash="dot",
                line_color="teal",
                opacity=0.5,
                annotation_text="IAR Upper",
                annotation_position="right",
                secondary_y=False
            )
        
        # Get replacement points
        model_override = eval_result.get('model_override', False)
        override_indices = eval_result.get('override_indices', [])
        
        # Find replacements
        replacement_timesteps = [t for t, a in zip(timesteps, actions) if a == 0]
        
        # Separate normal and override replacements
        if model_override and override_indices:
            override_replacements = [t for t in replacement_timesteps if t in override_indices]
            normal_replacements = [t for t in replacement_timesteps if t not in override_indices]
        else:
            normal_replacements = replacement_timesteps
            override_replacements = []
        
        # Add normal replacements as red markers
        if normal_replacements:
            normal_wear_values = [tool_wear[timesteps.index(t)] - 5 for t in normal_replacements]
            fig.add_trace(
                go.Scatter(
                    x=normal_replacements,
                    y=normal_wear_values,
                    name="Tool Replacement",
                    mode='markers',
                    marker=dict(
                        size=12,
                        color='#EF553B',
                        symbol='diamond',
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                secondary_y=False
            )
        
        # Add overridden replacements if any
        if override_replacements:
            override_wear_values = [tool_wear[timesteps.index(t)] - 5 for t in override_replacements]
            fig.add_trace(
                go.Scatter(
                    x=override_replacements,
                    y=override_wear_values,
                    name="Tool-Replacements",
                    mode='markers',
                    marker=dict(
                        size=12,
                        color="#EF553B",
                        symbol='diamond',
                        opacity=0.7
                    ),
                    showlegend=True
                ),
                secondary_y=False
            )
        
        # Update layout
        fig.update_xaxes(title_text="Timestep")
        fig.update_yaxes(title_text="Tool Wear", secondary_y=False)
        fig.update_yaxes(title_text="Action", secondary_y=True, range=[0.0, 1.5], showticklabels=False, ticks="", showgrid=False)
        
        fig.update_layout(
            title=f"Model Evaluation: {model_filename}_{test_filename}",
            height=500,
            template="plotly_white",
            plot_bgcolor='#f0f2f6',
            hovermode='x unified'
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white', secondary_y=False)
        
        # Save plot
        plot_filename = f"{model_filename}_Eval_{test_filename}.png"
        plot_filepath = os.path.join(results_dir, plot_filename)
        
        fig.write_image(plot_filepath, width=1200, height=600)
        
        return plot_filepath
        
    except Exception as e:
        print(f"  [!] Could not create plot: {str(e)}")
        return None


def _build_eval_row(algo, att_short, att_full_name, lr, gm, model_filename,
                    train_filename, test_filename, self_eval, eval_result,
                    eval_round: int, retry_index: int) -> Dict:
    """Helper: build a single result row dict from an eval_result."""
    return {
        'Round': eval_round,
        'Retry Index': retry_index,
        'Algorithm': f"{algo}",
        'Attention Mechanism': att_full_name,
        'Learning Rate': lr if lr is not None else "N/A",
        'Gamma': gm if gm is not None else "N/A",
        'Model File': model_filename,
        'Training File': train_filename,
        'Test File': test_filename,
        'Self-eval': self_eval,
        'Tool Usage %': eval_result.get('tool_usage_pct', 0.0) * 100 if eval_result.get('tool_usage_pct') else 0,
        'Lambda': eval_result.get('lambda', 0),
        'Threshold Violations': eval_result.get('threshold_violations', 0),
        'T_wt': eval_result.get('T_wt'),
        't_FR': eval_result.get('t_FR'),
        'Eval_Score': calculate_eval_score(eval_result),
    }


def evaluate_agents(schema: str, trained_models: Dict, skip_individual_plots: bool = False,
                    num_eval_rounds: int = 1) -> pd.DataFrame:
    """
    Evaluate all trained models on all data files of the schema.

    Each (model × test_file) combination is evaluated num_eval_rounds times.  Every
    round produces one row in the returned DataFrame (Round = 1..num_eval_rounds).
    This allows proper statistical hypothesis testing (e.g. t-test) across algorithms.

    Args:
        schema: 'SIT' or 'IEEE'
        trained_models: Dict mapping (algo, training_file, attention_type, lr, gamma) -> model_path
        skip_individual_plots: If True, skip per-model evaluation plots
        num_eval_rounds: Number of evaluation repetitions per (model × test_file) combo.
                         Use 1 for single-round (default); EVAL_ROUNDS for multi-round.

    Returns:
        DataFrame with num_eval_rounds rows per (model × test_file) combination.
        Plots (heatmap, analysis) should aggregate (mean) across rounds.
    """

    # Checkpoint Model Evaluation - recovery from failuer - Maximum retries
    # RETRY_EVAL = 10
    # EVAL_ROUNDS = 20  # Number of evaluation rounds for multi-round evaluation (if -V N specified)
    RETRY_THRESHOLDS = {'A2C':  0.70, 'DQN':  0.70, 'PPO':  0.85}
    
    is_multi = num_eval_rounds > 1
    print(f"\n{'='*80}")
    if is_multi:
        print(f"EVALUATION PHASE: MULTI-ROUND | {num_eval_rounds} rounds per combo | "
              f"{len(trained_models)} model(s)")
    else:
        print(f"EVALUATION PHASE: Single-round | {len(trained_models)} model(s)")
    print(f"{'='*80}\n")

    # Attention short → full name map
    attention_map = {
        'MH': 'Multi-Head',
        'NW': 'Nadaraya-Watson',
        'SA': 'Self-Attention',
        'TP': 'Temporal',
    }

    # Get test files
    test_files = get_schema_files(schema)

    results = []

    for model_key, model_path in trained_models.items():
        # Unpack model key
        if len(model_key) == 5:
            algo, train_filename, att_short, lr, gm = model_key
        elif len(model_key) == 3:
            algo, train_filename, att_short = model_key
            lr, gm = None, None
        else:
            print(f"  [!] Unexpected model key format: {model_key}, skipping...")
            continue

        att_label        = f" ({att_short})" if att_short else ""
        hyperparam_label = f" | LR={lr} | G={gm}" if lr is not None else ""
        att_full_name    = attention_map.get(att_short, att_short) if att_short else 'None'
        model_filename   = Path(model_path).stem

        print(f"\nEvaluating {algo}{att_label}{hyperparam_label} (trained on {train_filename}):")

        for test_file in test_files:
            test_filename = Path(test_file).stem
            self_eval     = 'Y' if train_filename == test_filename else 'N'

            # ── Inner loop: one row per round ──────────────────────────────────
            for round_i in range(1, num_eval_rounds + 1):
                if is_multi:
                    # Each round uses a distinct, spread seed for genuine variance.
                    # Single-round stays at seed=42 for backward compatibility.
                    base_seed = (round_i * 97 + 13) % 9973
                    print(f"  [Round {round_i}/{num_eval_rounds}] {test_filename} (seed={base_seed})")
                else:
                    base_seed = 42

                try:
                    # ── Initial evaluation for this round ─────────────────────
                    eval_result = rl_pdm.evaluate_trained_model(model_path, test_file, seed=base_seed)

                    if eval_result.get('error', False):
                        print(f"  [!] {test_filename} R{round_i}: Feature mismatch – skipping")
                        continue

                    initial_score    = calculate_eval_score(eval_result)
                    best_eval_result = eval_result
                    best_score       = initial_score
                    best_retry_idx   = 1

                    # ── Retry logic
                    if algo != 'REINFORCE' and algo in RETRY_THRESHOLDS:
                        threshold = RETRY_THRESHOLDS[algo]

                        if initial_score >= threshold:
                            print(f"    [~] R{round_i} Score {initial_score:.4f} >= {threshold:.2f} "
                                  f"– retrying up to {RETRY_EVAL - 1} more time(s)...")

                            for retry_i in range(2, RETRY_EVAL + 1):
                                try:
                                    # Seeds are both round- and retry-aware for maximum spread
                                    retry_seed = (round_i * 1000 + retry_i * 97 + 31) % 9973
                                    retry_result = rl_pdm.evaluate_trained_model(
                                        model_path, test_file, seed=retry_seed
                                    )
                                    if retry_result.get('error', False):
                                        continue
                                    retry_score = calculate_eval_score(retry_result)
                                    print(f"    [~] R{round_i} Retry {retry_i}/{RETRY_EVAL} "
                                          f"(seed={retry_seed}): Score={retry_score:.4f}")

                                    if retry_score < best_score:
                                        best_score       = retry_score
                                        best_eval_result = retry_result
                                        best_retry_idx   = retry_i

                                    if retry_score < threshold:
                                        print(f"    [✓] R{round_i}: {retry_score:.4f} < {threshold:.2f} "
                                              f"at retry {retry_i} – stopping.")
                                        break
                                except Exception as re_err:
                                    print(f"    [!] R{round_i} Retry {retry_i} error: {re_err}")

                            if best_score >= threshold:
                                print(f"    [⚠] R{round_i} RETRY EXHAUSTED – best {best_score:.4f} "
                                      f">= {threshold:.2f}. Saving minimum (retry={best_retry_idx}).")
                            else:
                                print(f"    [✓] R{round_i} Final: {best_score:.4f} (retry={best_retry_idx})")
                        else:
                            print(f"    [-] R{round_i} Score {initial_score:.4f} < {threshold:.2f} "
                                  f"– no retry needed.")
                    # REINFORCE: no retry, best_retry_idx stays 1

                    # ── Build row for this round ───────────────────────────────
                    row = _build_eval_row(
                        algo, att_short, att_full_name, lr, gm, model_filename,
                        train_filename, test_filename, self_eval,
                        best_eval_result, round_i, best_retry_idx
                    )

                    results.append(row)
                    print(f"  [+] R{round_i} {test_filename}: Lambda={row['Lambda']}, "
                          f"Violations={row['Threshold Violations']}, "
                          f"Score={row['Eval_Score']:.4f}, RetryIdx={row['Retry Index']}")

                    # ── Individual eval plot (single-round only, round 1 only) ──
                    if not skip_individual_plots and round_i == 1:
                        plot_path = create_evaluation_plot(best_eval_result, model_filename, test_filename)
                        if plot_path:
                            print(f"    Plot saved: {plot_path}")

                except Exception as e:
                    print(f"  [x] R{round_i} {test_filename}: Evaluation error – {str(e)}")

    results_df = pd.DataFrame(results)

    total_rounds = results_df['Round'].nunique() if not results_df.empty else 0
    print(f"\n{'='*80}")
    print(f"Evaluation complete. Rounds={total_rounds} | Total rows: {len(results_df)}")
    if is_multi:
        print(f"  (CSV retains all rows for statistical analysis, e.g. t-tests)")
        print(f"  (Heatmap and Analysis plots will use mean across rounds)")
    print(f"{'='*80}\n")

    return results_df


def calculate_eval_score(eval_result: Dict) -> float:
    """
    Calculate weighted evaluation score based on metrics.
    
    Weights:
    - Tool Usage: 50%
    - Lambda: 30%
    - Violations: 20%
    
    Args:
        eval_result: Dictionary from adjusted_evaluate_model
    
    Returns:
        Float score between 0-1 (higher is better)
    """
    tool_usage_pct = eval_result.get('tool_usage_pct', 0) or 0
    lambda_metric = eval_result.get('lambda', 0) or 0
    violations = eval_result.get('threshold_violations', 0)
    
    # Normalize tool usage (0-100% is ideal)
    tool_usage_score = min(1.0, max(0.0, tool_usage_pct))
    
    # Normalize lambda (lower is better, 0 is best)
    # Lambda max is the wear threshold - IAR_LOWER
    IAR_LOWER = rl_pdm.WEAR_THRESHOLD * (1.0 - rl_pdm.IAR_RANGE)
    lambda_score = max(0.0, 1.0 - (abs(lambda_metric) / IAR_LOWER))
    
    # Normalize violations (lower is better, 0 is best)
    violation_score = max(0.0, 1.0 / (1.0 + violations))
    
    # Weighted combination
    eval_score = (
        0.50 * tool_usage_score +
        0.30 * lambda_score +
        0.20 * violation_score
    )
    
    return eval_score


def save_results(results_df: pd.DataFrame, schema: str, attention_mech: int,
                 results_dir: str = "results", multiround: bool = False,
                 batch_idx: int = None) -> str:
    """
    Save evaluation results to CSV file.

    Args:
        results_df: DataFrame with results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save results
        multiround: If True, use 'Evaluation_Results_Multiround_' prefix
        batch_idx: Batch number for filename labelling (e.g. 1 -> B1)

    Returns:
        Path to saved CSV file
    """
    os.makedirs(results_dir, exist_ok=True)

    # Create filename with timestamp
    now = datetime.now()
    timestamp = now.strftime(timestamp_fmt)
    att_label = "AM" if attention_mech else "NoAM"
    batch_label = f"_B{batch_idx}" if batch_idx is not None else ""

    if multiround:
        filename = f"_Evals_Multiround_{schema}{batch_label}_{timestamp}.csv"
    else:
        filename = f"_Evals_Results_{schema}{batch_label}_{timestamp}.csv"

    filepath = os.path.join(results_dir, filename)

    # Save to CSV
    results_df.to_csv(filepath, index=False)
    print(f"\n✓ Results saved to: {filepath}")

    return filepath


def create_heatmaps(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results",
                    batch_idx: int = None, batch_files: List[str] = None):
    """
    Create a single comprehensive heatmap showing evaluation scores for all agents.
    
    Args:
        results_df: DataFrame with evaluation results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save heatmap
        batch_idx: Batch number for filename labelling (e.g. 1 -> B1)
        batch_files: List of training file paths used in this batch (for subtitle)
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Pivot for heatmap: rows = ALL model files (all agents), cols = test files
    # Use mean so multi-round results are averaged across rounds for the plot.
    heatmap_data = results_df.pivot_table(
        index='Model File',
        columns='Test File',
        values='Eval_Score',
        aggfunc='mean'
    )
    
    # Sort rows alphabetically for consistent ordering
    heatmap_data = heatmap_data.sort_index()
    # Build friendly labels for model rows by reading each model's _metadata.json file
    try:
        # Attention mechanism code mapping (metadata value -> short display code)
        att_code_map = {
            'multi_head':          'MH',
            'multi-head':          'MH',
            'multihead':           'MH',
            'nadaraya_watson':     'NW',
            'nadaraya-watson':     'NW',
            'nadaraya':            'NW',
            'nw':                  'NW',
            'self_attention':      'SA',
            'self-attention':      'SA',
            'selfattention':       'SA',
            'selfattn':            'SA',
            'temporal':            'TP',
            'tp':                  'TP',
            'sa':                  'SA',
            'mh':                  'MH',
        }

        att_display_name_map = {
            'MH': 'Multi-Head',
            'NW': 'Nadaraya-Watson',
            'SA': 'Self-Attention',
            'TP': 'Temporal',
        }

        def normalize_att_code(att_value) -> str:
            """Return a 2-letter attention code (e.g. 'MH') from metadata values."""
            if not att_value:
                return ''
            raw = str(att_value).strip()

            # If metadata already contains a short code (MH/NW/SA/TP), accept it.
            upper = raw.upper()
            if upper in att_display_name_map:
                return upper

            # Normalize common string variants.
            key = raw.lower().strip()
            key = re.sub(r"[\s\-]+", "_", key)
            return att_code_map.get(key, '')

        # Helper: format LR as e-notation without leading zeros (0.0005 -> 5e-4)
        def fmt_lr(x):
            if x is None:
                return None
            try:
                xv = float(x)
                s = "{:.0e}".format(xv)
                s = re.sub(r'e([+-])0+(\d+)', r'e\1\2', s)
                return s
            except Exception:
                return str(x)

        friendly_labels = []
        for model_file in heatmap_data.index:
            meta = {}
            meta_path = os.path.join("models", f"{model_file}_metadata.json")
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception:
                pass  # If JSON missing, meta stays empty and we fall back gracefully

            algo        = meta.get('algorithm', model_file)
            att_raw     = meta.get('attention_mechanism') or ''
            lr_raw      = meta.get('learning_rate')
            gm_raw      = meta.get('gamma')
            training    = str(meta.get('training_data', '')).replace('_', '-')

            # Resolve attention short code
            att_code = normalize_att_code(att_raw)

            # Resolve attention display name for label
            att_display = att_display_name_map.get(att_code, att_code) if att_code else ''

            lr_str = fmt_lr(lr_raw) if lr_raw is not None else None
            gm_str = str(gm_raw) if gm_raw is not None else None

            # Format: Algo [AM] LR Gamma Training
            parts = [str(algo)]
            if att_display:
                parts.append(att_display)
            if lr_str:
                parts.append(lr_str)
            if gm_str:
                parts.append(gm_str)
            if training:
                parts.append(training)

            label = ' '.join(parts)
            friendly_labels.append(label)

        # Replace heatmap row index with friendly labels for display
        heatmap_data.index = friendly_labels
    except Exception:
        # Fall back to original model file names if anything goes wrong
        pass
    
    # Create single comprehensive heatmap with larger figsize for all agents
    num_agents = len(heatmap_data)
    figsize_height = max(10, num_agents * 0.5)  # Scale height based on number of agents
    fig, ax = plt.subplots(figsize=(14, figsize_height))
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('#f0f2f6')
    
    # Use color scheme: Green (1.0), Yellow (0.5), Red (0.0)
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        cbar_kws={'label': 'Evaluation Score'},
        ax=ax,
        linewidths=0.5
    )
    
    batch_label = f"B{batch_idx}" if batch_idx is not None else ""
    file_subtitle = ", ".join([Path(f).stem for f in batch_files]) if batch_files else ""
    title_str = f'Evaluation Score Heatmap  |  Schema: {schema}'
    if batch_label:
        title_str += f'  |  {batch_label}'
    ax.set_title(title_str, fontsize=14, fontweight='bold')
    if file_subtitle:
        ax.set_xlabel(f'Test File\n{file_subtitle}', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('Test File', fontsize=12, fontweight='bold')
    ax.set_ylabel('RL PdM Model', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels (test files) to be horizontal for readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
    # Rotate y-axis labels (model files) to be horizontal for readability
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # Save single heatmap
    now = datetime.now()
    timestamp = now.strftime(timestamp_fmt)
    att_label = "AM" if attention_mech else "NoAM"
    batch_suffix = f"_B{batch_idx}" if batch_idx is not None else ""
    filename = f"_Heatmap_{schema}{batch_suffix}_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)
    
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    if PDF_REPORTS:
        pdf_path = filepath.replace('.png', '.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
        print(f"  ✓ Heatmap PDF saved: {pdf_path}")
    plt.close(fig)
    

def generate_analysis_report(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results",
                             batch_idx: int = None, batch_files: List[str] = None):
    """
    Generate a comprehensive analysis report with statistical plots and error bars.
    
    Creates a detailed 4-panel summary:
    1. Overall Performance Matrix (Heatmap)
    2. Model Performance by Attention (Grouped Bar Chart, vertical)
    3. Algorithm Performance (Bar Chart with 68% CI)
    4. Lambda Metric by Algorithm & Attention (Grouped Bar Chart)
    
    Args:
        results_df: DataFrame with evaluation results
        schema: 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag
        results_dir: Directory to save report
        batch_idx: Batch number for filename labelling (e.g. 1 -> B1)
        batch_files: List of training file paths used in this batch (for subtitle)
    """
    print(f"\nGenerating comprehensive analysis report...")
    os.makedirs(results_dir, exist_ok=True)
    
    # Check if 'Attention Mechanism' exists (it should, based on previous steps)
    if 'Attention Mechanism' not in results_df.columns:
        # Fallback if column missing (e.g. older version)
        results_df['Attention Mechanism'] = 'None'
    
    # Set plot style
    sns.set_theme(style="whitegrid")
    
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Grid: 2 rows, 2 columns
    # Row 1, Col 1: Overall Heatmap
    # Row 1, Col 2: Model Performance (Grouped)
    # Row 2, Col 1: Algo Performance
    # Row 2, Col 2: Attention Performance
    # Increased padding to prevent overlap
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.15)
    
    axs = []
    axs.append(fig.add_subplot(gs[0, 0])) # ax0: Heatmap
    axs.append(fig.add_subplot(gs[0, 1])) # ax1: Model Perf
    axs.append(fig.add_subplot(gs[1, 0])) # ax2: Algo Perf
    axs.append(fig.add_subplot(gs[1, 1])) # ax3: Attn Perf
    
    # Calculate dynamic axis limits
    min_score = results_df['Eval_Score'].min()
    max_score = results_df['Eval_Score'].max()
    
    # Heuristic: Lower bound = min_score - (range * 0.2) or min_score - 0.05
    # Ensure it doesn't go below 0
    # If all scores are high (e.g. > 0.9), we want to zoom in
    padding = max(0.05, (max_score - min_score) * 0.2)
    lower_bound = max(0.0, min_score - padding)
    # Round down to nearest 0.05
    lower_bound = np.floor(lower_bound * 20) / 20.0
    
    upper_bound = 1.0
    
    print(f"  Axis limits: [{lower_bound:.2f}, {upper_bound:.2f}] (Min score: {min_score:.4f})")

    # Canonical ordering for all plots
    ALGO_ORDER = ['A2C', 'DQN', 'PPO', 'REINFORCE']
    ATTN_ORDER = ['None', 'Nadaraya-Watson', 'Temporal', 'Self-Attention', 'Multi-Head']

    # --- Panel A: Overall Performance (Heatmap) ---
    ax = axs[0]
    try:
        # Aggregation
        pivot_data = results_df.pivot_table(
            index='Algorithm', 
            columns='Attention Mechanism', 
            values='Eval_Score', 
            aggfunc='mean'
        )
        # Reorder rows/cols
        pivot_rows = [r for r in ALGO_ORDER if r in pivot_data.index]
        pivot_cols = [c for c in ATTN_ORDER if c in pivot_data.columns]
        pivot_data = pivot_data.loc[pivot_rows, pivot_cols]
        
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt='.3f', 
            cmap='RdYlGn', 
            vmin=lower_bound, 
            vmax=upper_bound, 
            cbar_kws={'label': 'Avg Eval Score'}, 
            ax=ax,
            linewidths=0.2,
            # linecolor='grey'
        )
        ax.set_title('A. OVERALL PERFORMANCE (Avg Score)', fontsize=14, fontweight='bold', loc='left', pad=15)
        ax.set_xlabel('Attention Mechanism', fontweight='bold')
        ax.set_ylabel('Algorithm', fontweight='bold')
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate heatmap: {e}", ha='center')

    # --- Panel B: Model Performance (Grouped Bar with Error Bars) ---
    ax = axs[1]
    try:
        # Vertical Grouped Bar Chart
        # x = Attention, y = Score, hue = Algo
        sns.barplot(
            data=results_df,
            x='Attention Mechanism',
            y='Eval_Score',
            hue='Algorithm',
            order=[o for o in ATTN_ORDER if o in results_df['Attention Mechanism'].values],
            hue_order=[o for o in ALGO_ORDER if o in results_df['Algorithm'].values],
            errorbar=('ci', 68),  # 1-sigma (68%) confidence interval
            palette='viridis',
            ax=ax,
            capsize=0.1
        )
        ax.set_title('B. MODEL PERFORMANCE (by Attention & Algo)', fontsize=14, fontweight='bold', loc='left', pad=15)
        ax.set_xlabel('Attention Mechanism', fontweight='bold')
        ax.set_ylabel('Evaluation Score', fontweight='bold')
        ax.set_ylim(lower_bound, 1.02)  # Little bit of padding on top
        ax.legend(title='Algorithm', loc='lower right')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate model plot: {e}", ha='center')

    # --- Panel C: Algorithm Performance (Bar with Error Bars) ---
    ax = axs[2]
    try:
        sns.barplot(
            data=results_df,
            x='Algorithm',
            y='Eval_Score',
            order=[o for o in ALGO_ORDER if o in results_df['Algorithm'].values],
            errorbar=('ci', 68),  # 1-sigma (68%) confidence interval
            palette='Blues_d',
            ax=ax,
            capsize=0.1
        )
        ax.set_title('C. ALGORITHM PERFORMANCE (Avg ± CI)', fontsize=14, fontweight='bold', loc='left', pad=20)
        ax.set_xlabel('Algorithm', fontweight='bold')
        ax.set_ylabel('Evaluation Score', fontweight='bold')
        ax.set_ylim(lower_bound, 1.02)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate algo plot: {e}", ha='center')

    # --- Panel D: Lambda Metric by Algorithm & Attention (Grouped Bar) ---
    ax = axs[3]
    try:
        sns.barplot(
            data=results_df,
            x='Attention Mechanism',
            y='Lambda',
            hue='Algorithm',
            order=[o for o in ATTN_ORDER if o in results_df['Attention Mechanism'].values],
            hue_order=[o for o in ALGO_ORDER if o in results_df['Algorithm'].values],
            errorbar=('ci', 68),  # 1-sigma (68%) confidence interval
            palette='viridis',
            ax=ax,
            capsize=0.1
        )
        ax.set_title('D. LAMBDA METRIC (by Attention & Algo)', fontsize=14, fontweight='bold', loc='left', pad=15)
        ax.set_xlabel('Attention Mechanism', fontweight='bold')
        ax.set_ylabel('Lambda (Avg ± CI 68%)', fontweight='bold')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend(title='Algorithm', loc='upper right')
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not generate lambda plot: {e}", ha='center')

    # Main Title + subtitle with batch info and files
    batch_label = f"B{batch_idx}" if batch_idx is not None else ""
    main_title = f"AutoRL Evaluation Analysis Report | Schema: {schema}"
    if batch_label:
        main_title += f" | {batch_label}"
    plt.suptitle(main_title, fontsize=20, fontweight='bold', y=0.99)
    if batch_files:
        file_subtitle = ", ".join([Path(f).stem for f in batch_files])
        plt.figtext(0.5, 0.965, file_subtitle, ha='center', fontsize=11, color='#444444')

    # Save
    now = datetime.now()
    timestamp = now.strftime(timestamp_fmt)
    att_label = "AM" if attention_mech else "NoAM"
    batch_suffix = f"_B{batch_idx}" if batch_idx is not None else ""
    filename = f"_Analysis_Report_{schema}{batch_suffix}_{timestamp}.png"
    filepath = os.path.join(results_dir, filename)
    
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Analysis report saved: {filepath}")
        if PDF_REPORTS:
            pdf_path = filepath.replace('.png', '.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            print(f"  ✓ Analysis report PDF saved: {pdf_path}")
    except Exception as e:
        print(f"  [!] Failed to save report: {e}")
    finally:
        plt.close(fig)


def generate_statistical_analysis(results_df: pd.DataFrame, schema: str, attention_mech: int, results_dir: str = "results"):
    """
    Generate a statistical hypothesis testing report: REINFORCE vs A2C, DQN, PPO.

    Uses Welch's independent two-sample t-test (one-tailed, H₁: REINFORCE better than competitor).
    Metric direction is handled correctly:
      - Higher is better : Eval_Score, Tool Usage %   → H₁: REINFORCE > competitor  (t > 0 = good)
      - Lower  is better : Lambda, Threshold Violations → t is NEGATED so positive t = REINFORCE better
    Produces a 2×2 grid of t-statistic heatmaps:
      - Rows    : With self-eval (full dataset) | Without self-eval (unseen data only)
      - Columns : All attention mechanisms      | REINFORCE + best attention vs others

    Each cell (competitor × metric) shows:
      t-statistic (normalised), one-tailed p-value, significance stars.
      Gold border = |t| ≥ 1.96 (statistically significant at α = 0.05).
      Green    = REINFORCE better   |   Red = competitor better.

    Args:
        results_df    : DataFrame with evaluation results (all rounds)
        schema        : 'SIT' or 'IEEE'
        attention_mech: Attention mechanism flag (used only for filename labelling)
        results_dir   : Output directory
    """
    try:
        from scipy import stats as sp_stats
    except ImportError:
        print("  [!] scipy not found - skipping statistical analysis. Install: pip install scipy")
        return

    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    print(f"\nGenerating hypothesis testing / statistical analysis report...")
    os.makedirs(results_dir, exist_ok=True)

    # ── Constants ─────────────────────────────────────────────────────────────
    COMPETITORS   = ['A2C', 'DQN', 'PPO']
    METRICS       = ['Eval_Score', 'Tool Usage %', 'Lambda', 'Threshold Violations']
    METRIC_LABELS = ['Eval Score', 'Tool Usage %', 'Lambda', 'Violations']

    # ── Identify best attention mechanism for REINFORCE ───────────────────────
    rf_all = results_df[results_df['Algorithm'] == 'REINFORCE'].copy()
    if rf_all.empty:
        print("  [!] No REINFORCE rows found – skipping statistical analysis.")
        return

    if 'Attention Mechanism' in rf_all.columns and rf_all['Attention Mechanism'].nunique() > 0:
        attn_means = rf_all.groupby('Attention Mechanism')['Eval_Score'].mean()
        best_attn  = attn_means.idxmax()
        best_score = attn_means[best_attn]
    else:
        best_attn  = 'N/A'
        best_score = rf_all['Eval_Score'].mean()

    print(f"  Best attention for REINFORCE: {best_attn}  (mean Eval_Score = {best_score:.4f})")

    # ── Helper: compute t-stat / p-value matrices ─────────────────────────────
    # Metrics where LOWER values are better; t is negated so that
    # positive t always means "REINFORCE is better" across all metrics.
    LOWER_IS_BETTER = {'Lambda', 'Threshold Violations'}

    def tstat_matrix(df, use_best_attn=False):
        """
        Returns (t_mat, p_mat, n_rf_mat, n_comp_mat) each of shape (len(COMPETITORS), len(METRICS)).
        p_mat contains one-tailed p-values for H₁: REINFORCE better than competitor.
        n_rf_mat / n_comp_mat record the sample sizes used for each cell.

        For higher-is-better metrics (Eval_Score, Tool Usage %): REINFORCE wins when t > 0.
        For lower-is-better metrics (Lambda, Threshold Violations): t is NEGATED so that
        a positive value still means REINFORCE is better (i.e. REINFORCE has lower value).
        """
        if use_best_attn and 'Attention Mechanism' in df.columns:
            rf = df[(df['Algorithm'] == 'REINFORCE') & (df['Attention Mechanism'] == best_attn)]
        else:
            rf = df[df['Algorithm'] == 'REINFORCE']

        t_mat      = np.full((len(COMPETITORS), len(METRICS)), np.nan)
        p_mat      = np.ones ((len(COMPETITORS), len(METRICS)))
        n_rf_mat   = np.zeros((len(COMPETITORS), len(METRICS)), dtype=int)
        n_comp_mat = np.zeros((len(COMPETITORS), len(METRICS)), dtype=int)

        for i, algo in enumerate(COMPETITORS):
            comp = df[df['Algorithm'] == algo]
            for j, metric in enumerate(METRICS):
                r_vals = rf[metric].dropna().values
                c_vals = comp[metric].dropna().values
                n_rf_mat[i, j]   = len(r_vals)
                n_comp_mat[i, j] = len(c_vals)
                if len(r_vals) >= 2 and len(c_vals) >= 2:
                    t, p2 = sp_stats.ttest_ind(r_vals, c_vals, equal_var=False)
                    # For lower-is-better metrics, negate t so positive = REINFORCE better
                    if metric in LOWER_IS_BETTER:
                        t = -t
                    # One-tailed: H₁: REINFORCE better than competitor (positive t after normalisation)
                    p1 = p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0
                    t_mat[i, j] = round(t, 4)
                    p_mat[i, j] = round(p1, 4)
        return t_mat, p_mat, n_rf_mat, n_comp_mat

    def sig_stars(p):
        if np.isnan(p) or p >= 0.05: return 'ns'
        elif p < 0.001: return '***'
        elif p < 0.01:  return '**'
        else:           return '*'

    # ── Build four panel scenarios ─────────────────────────────────────────────
    df_full   = results_df
    df_unseen = results_df[results_df['Self-eval'] == 'N']

    panels_def = [
        (df_full,   False, 'A.  All Attention Mechanisms'),
        (df_full,   True,  f'B.  REINFORCE: {best_attn}  vs  Others (any attention)'),
        (df_unseen, False, 'C.  All Attention Mechanisms'),
        (df_unseen, True,  f'D.  REINFORCE: {best_attn}  vs  Others (any attention)'),
    ]

    panels = []
    for df, use_best, title in panels_def:
        if df.empty:
            t_mat      = np.zeros((len(COMPETITORS), len(METRICS)))
            p_mat      = np.ones ((len(COMPETITORS), len(METRICS)))
            n_rf_mat   = np.zeros((len(COMPETITORS), len(METRICS)), dtype=int)
            n_comp_mat = np.zeros((len(COMPETITORS), len(METRICS)), dtype=int)
        else:
            t_mat, p_mat, n_rf_mat, n_comp_mat = tstat_matrix(df, use_best)
        # Build a concise "(Sample sizes: N/M)" label for this panel.
        # N = REINFORCE sample count (same across competitors; use first competitor / Eval_Score).
        # M = competitor sample count: show a range if competitors differ, else a single number.
        n_rf_ref   = int(n_rf_mat[0, 0])                    # REINFORCE N (Eval_Score reference)
        n_comp_ref = n_comp_mat[:, 0].astype(int)           # competitor Ns for Eval_Score
        if n_comp_ref.min() == n_comp_ref.max():
            m_label = str(n_comp_ref[0])
        else:
            m_label = f"{n_comp_ref.min()}–{n_comp_ref.max()}"
        title_with_n = f"{title}  (Sample sizes: {n_rf_ref}/{m_label})"
        panels.append((t_mat, p_mat, n_rf_mat, n_comp_mat, title_with_n))

    # ── ASCII / Markdown / CSV output ─────────────────────────────────────────
    panel_short_names = [
        'A. All Attn - With Self-eval',
        f'B. Best Attn ({best_attn}) - With Self-eval',
        'C. All Attn - Unseen Only',
        f'D. Best Attn ({best_attn}) - Unseen Only',
    ]

    rows = []
    for (t_mat, p_mat, n_rf_mat, n_comp_mat, _), panel_name in zip(panels, panel_short_names):
        for i, competitor in enumerate(COMPETITORS):
            for j, (metric, mlabel) in enumerate(zip(METRICS, METRIC_LABELS)):
                t_val = t_mat[i, j]
                p_val = p_mat[i, j]
                stars = sig_stars(p_val)
                sig   = 'YES' if (not np.isnan(p_val) and p_val < 0.05) else 'no'
                rows.append({
                    'Panel'                 : panel_name,
                    'Competitor'            : competitor,
                    'Metric'               : mlabel,
                    'T-Statistic'          : f'{t_val:+.4f}' if not np.isnan(t_val) else 'N/A',
                    'P-Value (1-tail)'     : f'{p_val:.4f}'  if not np.isnan(p_val) else 'N/A',
                    'Significance'         : stars,
                    'Sig (α=0.05)'         : sig,
                    'REINFORCE wins?'      : ('Yes' if (not np.isnan(t_val) and t_val > 0 and not np.isnan(p_val) and p_val < 0.05) else '—'),
                    'REINFORCE Sample Size': int(n_rf_mat[i, j]),
                    'Others Sample Size'   : int(n_comp_mat[i, j]),
                })

    stat_df = pd.DataFrame(rows)

    # ── 3. Save CSV ────────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime(timestamp_fmt)
    att_label  = 'AM' if attention_mech else 'NoAM'
    csv_path   = os.path.join(results_dir, f'_Hypothesis_Tests_{schema}_{timestamp}.csv')
    stat_df.to_csv(csv_path, index=False)
    print(f'  ✓ Hypothesis test results CSV saved: {csv_path}\n')

    # Global symmetric colour scale (for t-stat plots)
    all_t    = np.concatenate([p[0].flatten() for p in panels])  # panels[i][0] is t_mat
    finite_t = all_t[np.isfinite(all_t)]
    t_abs_max = max(np.abs(finite_t).max() if len(finite_t) else 3.0, 2.0)
    t_abs_max = np.ceil(t_abs_max * 2) / 2.0   # round up to nearest 0.5

    # ── Helper: build one complete figure (t-stat OR p-value version) ──────────
    def _build_stat_figure(mode='t'):
        """
        mode='t'    → t-statistic heatmaps, symmetric RdYlGn scale.
        mode='pval' → p-value heatmaps, RdYlGn_r scale (green=significant=p~0).
        """
        sns.set_theme(style='white')

        # Taller figure + generous height ratios so banners never crush heatmaps
        fig = plt.figure(figsize=(24, 28))
        fig.patch.set_facecolor('#FAFBFF')

        # GridSpec: 6 rows × 2 cols
        #  row 0 : main title                         (generous height)
        #  row 1 : section banner 1 "WITH self-eval"  (visible height)
        #  row 2 : top heatmaps
        #  row 3 : section banner 2 "WITHOUT self-eval"
        #  row 4 : bottom heatmaps
        #  row 5 : colorbar + legend
        gs = fig.add_gridspec(
            6, 2,
            height_ratios=[0.55, 0.60, 3.0, 0.60, 3.0, 0.80],
            hspace=0.55, wspace=0.32,
            left=0.07, right=0.97, top=0.96, bottom=0.04
        )

        # ── Main title ─────────────────────────────────────────────────────────
        ax_title = fig.add_subplot(gs[0, :])
        ax_title.axis('off')

        if mode == 't':
            mode_label = "t-Statistic Scale"
            sub_desc   = (
                f"Welch's independent t-test (one-tailed, H1: REINFORCE better than competitor)  |  "
                f"Positive t = REINFORCE better  (Lambda & Violations: t negated; lower = better = green)  |  "
                f"Best attention for REINFORCE: {best_attn}  |  "
                f"Critical |t| ~1.96 (alpha=0.05)  |  "
                f"Generated: {datetime.now():%Y-%m-%d %H:%M}"
            )
        else:
            mode_label = "p-Value Scale"
            sub_desc   = (
                f"Welch's independent t-test (one-tailed, H1: REINFORCE better than competitor)  |  "
                f"Green = low p-value = statistically significant  |  "
                f"Best attention for REINFORCE: {best_attn}  |  "
                f"Gold border: p < 0.05 (alpha=0.05)  |  "
                f"Generated: {datetime.now():%Y-%m-%d %H:%M}"
            )

        ax_title.text(
            0.5, 0.92,
            f"Statistical Analysis - REINFORCE vs A2C, DQN, PPO  |  Schema: {schema}  |  {mode_label}",
            ha='center', va='top', fontsize=24, fontweight='bold', color='#1a1a2e',
            transform=ax_title.transAxes
        )
        ax_title.text(
            0.5, 0.32,
            sub_desc,
            ha='center', va='top', fontsize=16, color='#55557f',
            transform=ax_title.transAxes
        )

        # ── Section banners ────────────────────────────────────────────────────
        banner_cfg = [
            (1, "#476CD3",
             "[1] INCLUDING SELF-EVALUATION  (Full Dataset - trained files + unseen files)",
             "Left panel: all attention mechanisms.   "
             "Right panel: REINFORCE restricted to best attention; competitors unrestricted."),
            (3, "#476CD3",
             "[2] EXCLUDING SELF-EVALUATION  (Unseen Data Only - generalisation test)",
             "Same structure - only rows where Self-eval = N.   "
             "Tests whether REINFORCE generalises better than other algorithms."),
        ]
        for row_idx, bg, main_text, sub_text in banner_cfg:
            ax_b = fig.add_subplot(gs[row_idx, :])
            ax_b.axis('off')
            ax_b.set_facecolor(bg)
            ax_b.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_b.transAxes, color=bg, zorder=0))
            ax_b.text(0.015, 0.78, main_text, ha='left', va='top',
                      fontsize=16, fontweight='bold', color='white',
                      transform=ax_b.transAxes)
            ax_b.text(0.015, 0.15, sub_text, ha='left', va='bottom',
                      fontsize=14, color='#aaccee', transform=ax_b.transAxes)

        # ── Draw the four heatmaps ──────────────────────────────────────────────
        positions = [(2, 0), (2, 1), (4, 0), (4, 1)]

        start_hex = "#0790DF"  # A shade of blue
        end_hex = "#F14260"    # A shade of redish-pink

        # 2. Create the custom colormap
        # The 'colors' list defines the gradient stops.
        # You can include more colors for a multi-stop gradient (e.g., [start, middle, end]).
        custom_colors = [start_hex, end_hex]
        cmap_name = 'custom_gradient'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, custom_colors, N=256) # N defines the number of color bins for smooth gradient



        if mode == 't':
            cmap_name = 'RdYlGn'      # Berlin red=neg(bad), yellow=neutral, green=pos(REINFORCE better)
        else:
            # p-value: green=low p (significant), red=high p (not significant)
            # RdYlGn_r: vmin→green, vmax→red  →  p=0.000 is bright green
            cmap_name = custom_cmap

        for (gs_row, gs_col), (t_mat, p_mat, n_rf_mat, n_comp_mat, panel_title) in zip(positions, panels):
            ax = fig.add_subplot(gs[gs_row, gs_col])

            if mode == 't':
                display_mat = np.where(np.isnan(t_mat), 0.0, t_mat)
                vmin, vmax, center = -t_abs_max, t_abs_max, 0

                annot = np.empty_like(t_mat, dtype=object)
                for i in range(len(COMPETITORS)):
                    for j in range(len(METRICS)):
                        if np.isnan(t_mat[i, j]):
                            annot[i, j] = 'N/A'
                        else:
                            stars = sig_stars(p_mat[i, j])
                            annot[i, j] = f"t = {t_mat[i, j]:+.2f}\np = {p_mat[i, j]:.3f}  {stars}"
            else:
                # p-value mode: clip to [0,1]; NaN → 1.0 (white/red = not significant)
                display_mat = np.where(np.isnan(p_mat), 1.0, np.clip(p_mat, 0.0, 1.0))
                vmin, vmax, center = 0.0, 1.0, None

                annot = np.empty_like(p_mat, dtype=object)
                for i in range(len(COMPETITORS)):
                    for j in range(len(METRICS)):
                        if np.isnan(p_mat[i, j]):
                            annot[i, j] = 'N/A'
                        else:
                            stars = sig_stars(p_mat[i, j])
                            annot[i, j] = f"p = {p_mat[i, j]:.3f}\n{stars}"

            hm_kwargs = dict(
                annot=annot,
                fmt='',
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax,
                linewidths=0.25,
                ax=ax,
                cbar=False,
                xticklabels=METRIC_LABELS,
                yticklabels=COMPETITORS,
                annot_kws={'size': 13, 'va': 'center', 'color': '#1a1a1a', 'fontweight': 'bold'},
            )
            if center is not None:
                hm_kwargs['center'] = center
            sns.heatmap(display_mat, **hm_kwargs)

            # Gold border for significant cells
            for i in range(len(COMPETITORS)):
                for j in range(len(METRICS)):
                    if mode == 't':
                        highlight = (not np.isnan(t_mat[i, j]) and abs(t_mat[i, j]) >= 1.96)
                    else:
                        highlight = (not np.isnan(p_mat[i, j]) and p_mat[i, j] < 0.05)
                    if highlight:
                        ax.add_patch(plt.Rectangle(
                            (j, i), 1, 1,
                            fill=False, edgecolor="#A8F239", lw=2.0, zorder=4
                        ))

            ax.set_title(panel_title, fontsize=16, fontweight='bold',
                         loc='left', pad=18, color='#1a1a2e')
            ax.set_xlabel('Metric', fontsize=14, fontweight='bold', labelpad=10)
            ax.set_ylabel('Competitor Algorithm', fontsize=14, fontweight='bold', labelpad=10)
            ax.tick_params(axis='x', labelsize=14, rotation=20)
            ax.tick_params(axis='y', labelsize=15, rotation=0)

        # ── Shared colorbar + legend ───────────────────────────────────────────
        ax_cb = fig.add_subplot(gs[5, :])
        ax_cb.axis('off')

        if mode == 't':
            norm = mcolors.Normalize(vmin=-t_abs_max, vmax=t_abs_max)
            sm   = mcm.ScalarMappable(cmap=cmap_name, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax_cb, orientation='horizontal',
                              fraction=0.40, pad=0.04, aspect=50, shrink=0.85)
            cb.set_label(
                't-statistic  (positive = REINFORCE better; Lambda & Violations t is negated so lower value = positive)',
                fontsize=13, labelpad=8)
            cb.ax.tick_params(labelsize=12)
            for tv, col, ls in [(-1.96, '#FFD700', '--'), (0.0, '#444', '-'), (1.96, '#FFD700', '--')]:
                cb.ax.axvline(x=tv, color=col, lw=1.8, linestyle=ls)
            legend_text = (
                "Significance:  *** p < 0.001   |   ** p < 0.01   |   * p < 0.05   |   ns = not significant   ||   "
                "Gold border: |t| >= 1.96 (alpha=0.05)   ||   "
                "Positive t = REINFORCE BETTER than competitor   "
                "(for Lambda & Violations t is negated: lower value = better = positive t)"
            )
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            sm   = mcm.ScalarMappable(cmap=cmap_name, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax_cb, orientation='horizontal',
                              fraction=0.40, pad=0.04, aspect=50, shrink=0.85)
            cb.set_label('p-value  (green = significant = low p; red = not significant = high p)',
                         fontsize=13, labelpad=8)
            cb.ax.tick_params(labelsize=12)
            # Mark alpha=0.05 threshold on colorbar
            cb.ax.axvline(x=0.05, color='#FFD700', lw=2.0, linestyle='--')
            cb.ax.axvline(x=0.50, color='#888888', lw=1.2, linestyle=':')
            legend_text = (
                "Significance:  *** p < 0.001   |   ** p < 0.01   |   * p < 0.05   |   ns = not significant   ||   "
                "Gold border: p < 0.05 (alpha=0.05)   ||   "
                "Bright green = p ≈ 0.000 (highly significant)   |   Red = p ≈ 1.000 (not significant)"
            )

        ax_cb.text(
            0.5, 0.02, legend_text,
            ha='center', va='bottom', fontsize=12, color='#333355',
            transform=ax_cb.transAxes
        )

        return fig

    # ── Build & save both figures ──────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    att_label = "AM" if attention_mech else "NoAM"
    saved_paths = []

    for mode, name_key in [('t', '_Statistical_t'), ('pval', '_Statistical_p-value')]:
        fig = _build_stat_figure(mode=mode)
        filename = f"{name_key}_{schema}_{timestamp}.png"
        filepath = os.path.join(results_dir, filename)
        try:
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"  ✓ Statistical analysis saved: {filepath}")
            saved_paths.append(filepath)
            if PDF_REPORTS:
                pdf_path = filepath.replace('.png', '.pdf')
                plt.savefig(pdf_path, bbox_inches='tight', facecolor=fig.get_facecolor())
                print(f"  ✓ Statistical analysis PDF saved: {pdf_path}")
        except Exception as e:
            print(f"  [!] Failed to save {filename}: {e}")
        finally:
            plt.close(fig)

    return saved_paths[0] if saved_paths else None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRL: Train and Evaluate RL Agents for Predictive Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_agent.py -S SIT -A PPO,A2C -E 200 -AM 0
  python train_agent.py -S IEEE -A PPO -E 300 -AM 1
  python train_agent.py -S IEEE -A PPO -E 300 -AM 'NW', 'TP', 'MH', 'SA'
  python train_agent.py -S SIT -A PPO -E 200 -LR 0.001,0.0001 -G 0.95,0.99  # Grid search
  python train_agent.py -V -S IEEE  # Evaluate IEEE models
  python train_agent.py -D -S SIT   # Just list SIT models
  python train_agent.py  # Uses all defaults: SIT, PPO, 200 episodes, no attention
        """
    )
    
    parser.add_argument(
        '-S', '--schema',
        default='SIT',
        choices=['SIT', 'IEEE'],
        help="Data schema: 'SIT' or 'IEEE' (default: SIT)"
    )
    
    parser.add_argument(
        '-A', '--algos',
        default='PPO',
        help="Comma-separated list of algorithms: PPO,A2C,DQN,REINFORCE (default: PPO)"
    )
    
    parser.add_argument(
        '-E', '--episodes',
        type=int,
        default=200,
        help="Number of training episodes (default: 200)"
    )
    
    parser.add_argument(
        '-AM', '--attention-mechanism',
        type=str,
        default='0',
        choices=['0', '1', 'NW', 'TP', 'MH', 'SA'],
        help="Attention mechanism: 0=off, 1=all four types (NW/TP/MH/SA), or specify one type (default: 0)"
    )
    
    parser.add_argument(
        '-LR', '--learning-rates',
        type=str,
        default='0.0001',
        help="Comma-separated learning rate values for grid search (e.g., '0.001,0.0001,0.0005'). Default: 0.0001"
    )
    
    parser.add_argument(
        '-G', '--gamma',
        type=str,
        default='0.99',
        help="Comma-separated gamma values for grid search (e.g., '0.95,0.99'). Default: 0.99"
    )
    
    parser.add_argument(
        '-V', '--eval-only',
        nargs='?',
        type=int,
        const=0,
        default=None,
        metavar='N',
        help=(
            "Evaluation-only mode (skips training, uses discovered models). "
            "N=0: run eval + all plots (default when -V given with no number). "
            "N=1: run single-round eval, save only 3 main reports (no per-model plots). "
            f"N>1: multi-round mode – each (model x test_file) combo is evaluated {EVAL_ROUNDS} times "
            "(controlled by EVAL_ROUNDS global). All rows saved to Evaluation_Results_Multiround_xxxxx.csv "
            "with Round and Retry Index columns. Heatmap/Analysis plots show mean across rounds."
        )
    )

    parser.add_argument(
        '-D', '--discover',
        action='store_true',
        default=False,
        help="Discover mode: Only list trained models for the schema and exit (default: False)"
    )
    
    args = parser.parse_args()
    
    # Parse algorithms
    algos = [algo.strip().upper() for algo in args.algos.split(',')]
    
    # Parse attention mechanism value
    VALID_AM_TYPES = ['NW', 'TP', 'MH', 'SA']
    am_value = args.attention_mechanism
    if am_value == '0':
        attention_mech = 0
        attention_types = [None]
    elif am_value == '1':
        attention_mech = 1
        attention_types = VALID_AM_TYPES
    else:
        # Specific type e.g. 'MH', 'NW', 'TP', 'SA'
        attention_mech = 1
        attention_types = [am_value.upper()]
    
    # Validate algorithms
    valid_algos = ['PPO', 'A2C', 'DQN', 'REINFORCE']
    for algo in algos:
        if algo not in valid_algos:
            print(f"ERROR: Algorithm '{algo}' not supported. Choose from: {', '.join(valid_algos)}")
            sys.exit(1)
    
    # Parse learning rates
    if args.learning_rates:
        try:
            learning_rates = [float(lr.strip()) for lr in args.learning_rates.split(',')]
        except ValueError:
            print(f"ERROR: Invalid learning rate format. Use comma-separated float values (e.g., '0.001,0.0001')")
            sys.exit(1)
    else:
        learning_rates = [rl_pdm.LR_DEFAULT]
    
    # Parse gamma values
    if args.gamma:
        try:
            gammas = [float(g.strip()) for g in args.gamma.split(',')]
        except ValueError:
            print(f"ERROR: Invalid gamma format. Use comma-separated float values (e.g., '0.95,0.99')")
            sys.exit(1)
    else:
        gammas = [rl_pdm.GAMMA_DEFAULT]

    
    print(f"\n{'='*80}")
    print(f"AutoRL: RL Agent Training and Evaluation Pipeline")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Data Schema: {args.schema}")
    print(f"  Algorithms: {', '.join(algos)}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Learning Rates: {learning_rates}")
    print(f"  Gamma Values: {gammas}")
    if am_value == '0':
        am_display = 'OFF'
    elif am_value == '1':
        am_display = 'ALL (NW, TP, MH, SA)'
    else:
        am_display = f'Single type: {am_value.upper()}'
    print(f"  Attention Mechanism: {am_display}")
    
    # Calculate total training runs
    total_runs = len(algos) * len(learning_rates) * len(gammas)
    total_runs *= len(attention_types)  # 1 when off or single type, 4 when all
    print(f"  Total training runs (per file): {total_runs}")
    print(f"{'='*80}\n")
    
    
    try:
        # Phase 0: Initialize checkpoint database
        print(f"\n{'='*80}")
        print(f"CHECKPOINT SYSTEM: Initializing...")
        print(f"{'='*80}\n")
        init_checkpoint_db()
        print(f"✓ Checkpoint database initialized: {CHECKPOINT_DB}\n")
        
        # Phase 0.5: Check for incomplete batches (crash recovery)
        batch_id = None
        resume_mode = False
        
        if not args.discover and args.eval_only is None:
            incomplete_batch = get_incomplete_batch()
            if incomplete_batch:
                print(f"\n{'='*80}")
                print(f"⚠️  INCOMPLETE BATCH DETECTED")
                print(f"{'='*80}")
                print(f"  Batch ID: {incomplete_batch['batch_id']}")
                print(f"  Schema: {incomplete_batch['schema']}")
                print(f"  Algorithms: {incomplete_batch['algorithms']}")
                print(f"  Episodes: {incomplete_batch['episodes']}")
                print(f"  Status: {incomplete_batch['status']}")
                print(f"  Created: {incomplete_batch['created_at']}")
                print(f"{'='*80}\n")
                
                # Prompt user to resume
                try:
                    response = input("Resume this incomplete batch? (Y/n): ").strip().lower()
                    if response in ['', 'y', 'yes']:
                        batch_id = incomplete_batch['batch_id']
                        resume_mode = True
                        
                        # Load configuration from incomplete batch
                        args.schema = incomplete_batch['schema']
                        algos = incomplete_batch['algorithms'].split(',')
                        args.episodes = incomplete_batch['episodes']
                        learning_rates = [float(x) for x in incomplete_batch['learning_rates'].split(',')]
                        gammas = [float(x) for x in incomplete_batch['gammas'].split(',')]
                        attention_mech = incomplete_batch['attention_mech']
                        # Resume: pending tasks already have attention_type in DB; pass all types
                        # for context but resume_mode will rebuild queue from DB anyway
                        attention_types = VALID_AM_TYPES if attention_mech == 1 else [None]
                        
                        print(f"\n✓ Resuming batch: {batch_id}\n")
                        update_batch_status(batch_id, 'WIP')
                    else:
                        print(f"\n✓ Starting new batch (incomplete batch will remain in database)\n")
                except (EOFError, KeyboardInterrupt):
                    print(f"\n\n✓ Starting new batch\n")
        
        # Phase 0.6: Create new batch if not resuming
        if not resume_mode and not args.discover and args.eval_only is None:
            batch_id = create_batch_id()
            create_batch_record(
                batch_id=batch_id,
                schema=args.schema,
                algos=algos,
                episodes=args.episodes,
                lrs=learning_rates,
                gammas=gammas,
                attention_mech=attention_mech
            )
            update_batch_status(batch_id, 'WIP')
            print(f"✓ Created new batch: {batch_id}\n")
        
        # Phase 1: Discovery Only
        if args.discover:
            discover_trained_models(args.schema)
            sys.exit(0)

        # Phase 2: Training or Discovery
        if args.eval_only is not None:
            trained_models = discover_trained_models(args.schema)
        else:
            trained_models = train_agents(
                schema=args.schema,
                algos=algos,
                episodes=args.episodes,
                attention_mech=attention_mech,
                learning_rates=learning_rates,
                gammas=gammas,
                batch_id=batch_id,
                resume_mode=resume_mode,
                attention_types=attention_types
            )
        
        if not trained_models:
            print("ERROR: No models were successfully trained.")
            if batch_id:
                update_batch_status(batch_id, 'Failed')
            sys.exit(1)
        
        # Phase 3: Evaluation
        # ── Multi-round mode (any -V N where N > 1) ───────────────────────────────
        is_multiround = (args.eval_only is not None and args.eval_only > 1)

        if is_multiround:
            results_df = evaluate_agents(
                schema=args.schema,
                trained_models=trained_models,
                skip_individual_plots=not INDIVDUAL_PLOTS,
                num_eval_rounds=EVAL_ROUNDS,    # always use the global constant
            )
        # ── Single-round mode (N=0 or N=1) ───────────────────────────────────────
        else:
            results_df = evaluate_agents(
                schema=args.schema,
                trained_models=trained_models,
                skip_individual_plots=not INDIVDUAL_PLOTS,
                num_eval_rounds=1,
            )

        if results_df.empty:
            print("ERROR: Evaluation produced no results.")
            if batch_id:
                update_batch_status(batch_id, 'Failed')
            sys.exit(1)

        # Phase 4: Save results FIRST (all rows retained for statistical analysis)
        results_file = save_results(results_df, args.schema, attention_mech, multiround=is_multiround)

        # Phase 5: Create analysis report and heatmaps
        # Done AFTER the eval results file is written, using only final (lowest-score) rows.
        print(f"\nGenerating heatmaps from final evaluation results...")
        create_heatmaps(results_df, args.schema, attention_mech)

        print(f"\nGenerating analysis report from final evaluation results...")
        generate_analysis_report(results_df, args.schema, attention_mech)

        print(f"\nGenerating statistical analysis / hypothesis testing report...")
        generate_statistical_analysis(results_df, args.schema, attention_mech)

        # Phase 6: Mark batch as complete
        if batch_id:
            mark_batch_complete(batch_id)
            print(f"\n✓ Batch {batch_id} marked as COMPLETE\n")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*80}")
        print(f"Results Summary:")
        if is_multiround:
            print(f"  Evaluation Mode   : Multi-round ({EVAL_ROUNDS} rounds per combo)")
            print(f"  Total Rows Saved  : {len(results_df)}")
            print(f"  Unique Rounds     : {results_df['Round'].nunique()}")
            print(f"  (All rows retained in CSV for t-test / statistical analysis)")
            print(f"  (Heatmap and Analysis plots show mean across rounds)")
        else:
            print(f"  Evaluation Mode   : Single-round")
            print(f"  Total Evaluations : {len(results_df)}")
        print(f"  Models Evaluated  : {len(trained_models)}")
        print(f"  Test Files        : {results_df['Test File'].nunique()}")
        print(f"  Average Eval Score: {results_df['Eval_Score'].mean():.4f}")
        if batch_id:
            print(f"  Batch ID          : {batch_id}")
        print(f"\nResults saved in: results/")
        if is_multiround:
            print(f"Results file     : {results_file}")
        print(f"Checkpoint database: {CHECKPOINT_DB}")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Pipeline interrupted by user (Ctrl+C)")
        if batch_id:
            print(f"   Batch {batch_id} status: WIP (can be resumed)")
            print(f"   Run the script again to resume from where you left off.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Pipeline failed with error:")
        print(f"  {str(e)}")
        if batch_id:
            update_batch_status(batch_id, 'Failed')
            print(f"  Batch {batch_id} marked as FAILED")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
