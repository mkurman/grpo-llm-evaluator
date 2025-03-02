import gradio as gr
from grpo_llm_eval.config import TrainingConfig
import yaml
import subprocess
import json
import os
import pandas as pd
from glob import glob
from datetime import datetime

# Add custom CSS for styling
custom_css = """
body {
    font-family: Arial, sans-serif;
    color: #333;
}

.gradio-container {
    margin: 0 auto;
    padding: 20px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border-radius: 8px;
}

.gr-button {
    background-color: #007bff;
    color: #fff;
    margin: 5px;
    border: none;
    padding: 10px 20px;
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}

.gr-button:hover {
    background-color: #0056b3;
}

.gr-button img {
    margin-right: 8px;
}

.gr-textbox, .gr-number {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 4px;
    margin-bottom: 10px;
}

.gr-markdown {
    margin-bottom: 20px;
}

.gr-accordion {
    margin-bottom: 20px;
}

.gr-dataframe {
    margin-top: 20px;
}
"""


# Function to load the config
def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# Add this function after load_config():
def update_interface_from_config(config_path):
    try:
        config = load_config(config_path)
        return [
            config.get("openai_base_url", default_config.openai_base_url),
            config.get("student_model_name", default_config.student_model_name),
            config.get("teacher_model_name", default_config.teacher_model_name),
            config.get("dataset_name", default_config.dataset_name),
            config.get(
                "dataset_question_column", default_config.dataset_question_column
            ),
            config.get("dataset_answer_column", default_config.dataset_answer_column),
            config.get("output_dir", default_config.output_dir),
            config.get("save_steps", default_config.save_steps),
            config.get("learning_rate", default_config.learning_rate),
            config.get("max_new_tokens", default_config.max_new_tokens),
            config.get(
                "max_feedback_new_tokens", default_config.max_feedback_new_tokens
            ),
            config.get("num_return_sequences", default_config.num_return_sequences),
            config.get("accumulation_steps", default_config.accumulation_steps),
            config.get("temperature", default_config.temperature),
            config.get("top_p", default_config.top_p),
            config.get("top_k", default_config.top_k),
            config.get("max_seq_length", default_config.max_seq_length),
            config.get("cache_dir", default_config.cache_dir),
            config.get("warmup_steps", default_config.warmup_steps),
            config.get("total_steps", default_config.total_steps),
            config.get("seed", default_config.seed),
            config.get("max_grad_norm", default_config.max_grad_norm),
            config.get("grpo_beta", default_config.grpo_beta),
            config.get("sft_beta", default_config.sft_beta),
            config.get("thought_process_weight", default_config.thought_process_weight),
            config.get("answer_weight", default_config.answer_weight),
            config.get("format_weight", default_config.format_weight),
            config.get("system_prompt", default_config.system_prompt),
            config.get("evaluation_prompt", default_config.evaluation_prompt),
            config.get("think_open_string", default_config.think_open_string),
            config.get("think_close_string", default_config.think_close_string),
        ]
    except Exception as e:
        raise gr.Error(f"Error loading config: {str(e)}")


# Modify the run_training function
def run_training(*args, progress=gr.Progress()):
    """Get all arguments in order"""
    config_data = {
        "output_dir": args[0],
        "openai_base_url": args[1],
        "student_model_name": args[2],
        "teacher_model_name": args[3],
        "dataset_name": args[4],
        "dataset_question_column": args[5],
        "dataset_answer_column": args[6],
        "save_steps": args[7],
        "learning_rate": args[8],
        "max_new_tokens": args[9],
        "max_feedback_new_tokens": args[10],
        "num_return_sequences": args[11],
        "accumulation_steps": args[12],
        "temperature": args[13],
        "top_p": args[14],
        "top_k": args[15],
        "max_seq_length": args[16],
        "cache_dir": args[17],
        "warmup_steps": args[18],
        "total_steps": args[19],
        "seed": args[20],
        "max_grad_norm": args[21],
        "grpo_beta": args[22],
        "sft_beta": args[23],
        "thought_process_weight": args[24],
        "answer_weight": args[25],
        "format_weight": args[26],
        "system_prompt": args[27],
        "evaluation_prompt": args[28],
        "think_open_string": args[29],
        "think_close_string": args[30],
        "use_unsloth": args[31],
        "load_in_4bit": args[32],
    }

    output_dir = (
        f'{config_data["output_dir"]}/{datetime.now().strftime('%Y-%m-%d %H_%M_%S')}'
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Automatically save current config to the output directory
    config_path = os.path.join(output_dir, "training.yaml")

    config_data["output_dir"] = output_dir

    # Save config before training
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        print(f"Saved config to {config_path}")

    # Start training process
    print("Loading config from", config_path)
    command = ["grpo_llm_eval", "--config", config_path]
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    log_file = f"{output_dir}/training_log.txt"
    data_file = f"{output_dir}/responses.jsonl"

    print(f"Training log will be saved to {log_file}")
    print(f"Training data will be saved to {data_file}")

    last_size = 0

    with open(log_file, "w") as f:
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                print(line.strip())
                f.write(line)
                f.flush()

            # Check if responses.jsonl has been updated
            try:
                if os.path.exists(data_file):
                    current_size = os.path.getsize(data_file)
                    if current_size > last_size:
                        last_size = current_size
                        df = display_data(data_file)
                        yield line if line else "", df
                        continue
            except Exception:
                pass

            yield line if line else "", None

    process.wait()
    if process.returncode != 0:
        raise gr.Error(
            f"Training failed with return code {process.returncode}. Check {log_file} for details."
        )

    # Return final status and data
    final_df = display_data(data_file)
    yield f"Training completed. Check {log_file} for details.", final_df


# Function to display data
def display_data(data_file):
    """Load and display data from JSONL file using pandas"""
    try:
        # Read JSONL file directly into DataFrame
        df = pd.read_json(data_file, lines=True)
        df = df.sort_index(ascending=False)
        df.style.set_properties(**{"vertical-align": "text-top"})
        if df.empty:
            return pd.DataFrame({"error": ["No data found in results file"]})
        # Set proper DataFrame columns with descriptive names
        df.columns = [
            "Question",
            "Student Response",
            "Teacher Feedback",
            "Reward",
            "Thought Score",
            "Answer Score",
            "Style Score",
            "Advantage",
            "Policy Loss",
            "SFT Loss",
        ]
        return df
    except FileNotFoundError:
        return pd.DataFrame(
            {"error": ["No data found. Please ensure training has been run."]}
        )
    except ValueError as e:
        # Handle JSON parsing errors
        return pd.DataFrame({"error": [f"Error parsing results file: {str(e)}"]})
    except Exception as e:
        return pd.DataFrame({"error": [f"Unexpected error: {str(e)}"]})


# Get default values from TrainingConfig
default_config = TrainingConfig()

# Create the Gradio interface
with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("# GRPO LLM Evaluator Training Interface")
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### Configuration")
            config_path = gr.Textbox(label="Load from", value="configs/example.yaml")
            load_button = gr.Button("Load Configuration", variant="secondary")

            # Config file controls
            with gr.Group():

                # Configuration parameters
                with gr.Accordion("Settings", open=True):
                    openai_base_url = gr.Textbox(
                        label="OpenAI Base URL", value=default_config.openai_base_url
                    )
                    use_unsloth = gr.Checkbox(
                        label="Use Unsloth", value=default_config.use_unsloth
                    )
                    load_in_4bit = gr.Checkbox(
                        label="Load in 4-bit", value=default_config.load_in_4bit
                    )
                    student_model_name = gr.Textbox(
                        label="Student Model Name",
                        value=default_config.student_model_name,
                    )
                    teacher_model_name = gr.Textbox(
                        label="Teacher Model Name",
                        value=default_config.teacher_model_name,
                    )
                    dataset_name = gr.Textbox(
                        label="Dataset Name", value=default_config.dataset_name
                    )
                    dataset_question_column = gr.Textbox(
                        label="Dataset Question Column",
                        value=default_config.dataset_question_column,
                    )
                    dataset_answer_column = gr.Textbox(
                        label="Dataset Answer Column",
                        value=default_config.dataset_answer_column,
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory", value=default_config.output_dir
                    )
                    save_steps = gr.Number(
                        label="Save Steps", value=default_config.save_steps
                    )
                    learning_rate = gr.Number(
                        label="Learning Rate", value=default_config.learning_rate
                    )
                    max_new_tokens = gr.Number(
                        label="Max New Tokens", value=default_config.max_new_tokens
                    )
                    max_feedback_new_tokens = gr.Number(
                        label="Max Feedback New Tokens",
                        value=default_config.max_feedback_new_tokens,
                    )
                    num_return_sequences = gr.Number(
                        label="Num Return Sequences",
                        value=default_config.num_return_sequences,
                    )
                    accumulation_steps = gr.Number(
                        label="Accumulation Steps",
                        value=default_config.accumulation_steps,
                    )
                    temperature = gr.Number(
                        label="Temperature", value=default_config.temperature
                    )
                    top_p = gr.Number(label="Top P", value=default_config.top_p)
                    top_k = gr.Number(label="Top K", value=default_config.top_k)
                    max_seq_length = gr.Number(
                        label="Max Seq Length", value=default_config.max_seq_length
                    )
                    cache_dir = gr.Textbox(
                        label="Cache Directory", value=default_config.cache_dir
                    )
                    warmup_steps = gr.Number(
                        label="Warmup Steps", value=default_config.warmup_steps
                    )
                    total_steps = gr.Number(
                        label="Total Steps", value=default_config.total_steps
                    )
                    seed = gr.Number(label="Seed", value=default_config.seed)
                    max_grad_norm = gr.Number(
                        label="Max Grad Norm", value=default_config.max_grad_norm
                    )
                    grpo_beta = gr.Number(
                        label="GRPO Beta", value=default_config.grpo_beta
                    )
                    sft_beta = gr.Number(
                        label="SFT Beta", value=default_config.sft_beta
                    )
                    thought_process_weight = gr.Number(
                        label="Thought Process Weight",
                        value=default_config.thought_process_weight,
                    )
                    answer_weight = gr.Number(
                        label="Answer Weight", value=default_config.answer_weight
                    )
                    format_weight = gr.Number(
                        label="Format Weight", value=default_config.format_weight
                    )
                    system_prompt = gr.Textbox(
                        label="System Prompt", value=default_config.system_prompt
                    )
                    evaluation_prompt = gr.Textbox(
                        label="Evaluation Prompt",
                        value=default_config.evaluation_prompt,
                    )
                    think_open_string = gr.Textbox(
                        label="Think Open String",
                        value=default_config.think_open_string,
                    )
                    think_close_string = gr.Textbox(
                        label="Think Close String",
                        value=default_config.think_close_string,
                    )

        with gr.Column(scale=3, variant="panel"):
            # Button to run training
            run_button = gr.Button("Run Training", variant="primary")
            # Live output during training
            training_output = gr.Textbox(
                label="Training Output", lines=10, max_lines=15, autoscroll=True
            )
            # Display training results as DataFrame
            data_output = gr.DataFrame(
                label="Training Results",
                interactive=True,
                wrap=True,
                show_fullscreen_button=True,
                show_search=True,
                show_copy_button=True,
                headers=[
                    "Question",
                    "Student Response",
                    "Teacher Feedback",
                    "Reward",
                    "Thought Score",
                    "Answer Score",
                    "Style Score",
                    "Advantage",
                    "Policy Loss",
                    "SFT Loss",
                ],
            )

            # Update the run button click handler to include all inputs
            run_button.click(
                run_training,
                inputs=[
                    output_dir,
                    openai_base_url,
                    student_model_name,
                    teacher_model_name,
                    dataset_name,
                    dataset_question_column,
                    dataset_answer_column,
                    save_steps,
                    learning_rate,
                    max_new_tokens,
                    max_feedback_new_tokens,
                    num_return_sequences,
                    accumulation_steps,
                    temperature,
                    top_p,
                    top_k,
                    max_seq_length,
                    cache_dir,
                    warmup_steps,
                    total_steps,
                    seed,
                    max_grad_norm,
                    grpo_beta,
                    sft_beta,
                    thought_process_weight,
                    answer_weight,
                    format_weight,
                    system_prompt,
                    evaluation_prompt,
                    think_open_string,
                    think_close_string,
                    use_unsloth,
                    load_in_4bit,
                ],
                outputs=[training_output, data_output],
                show_progress=True,
            )

    # Add this after all the component declarations:
    load_button.click(
        update_interface_from_config,
        inputs=[config_path],
        outputs=[
            openai_base_url,
            student_model_name,
            teacher_model_name,
            dataset_name,
            dataset_question_column,
            dataset_answer_column,
            output_dir,
            save_steps,
            learning_rate,
            max_new_tokens,
            max_feedback_new_tokens,
            num_return_sequences,
            accumulation_steps,
            temperature,
            top_p,
            top_k,
            max_seq_length,
            cache_dir,
            warmup_steps,
            total_steps,
            seed,
            max_grad_norm,
            grpo_beta,
            sft_beta,
            thought_process_weight,
            answer_weight,
            format_weight,
            system_prompt,
            evaluation_prompt,
            think_open_string,
            think_close_string,
        ],
    )


# Launch the Gradio interface
def run_ui():
    """Entry point for the Gradio UI"""
    iface.launch()


if __name__ == "__main__":
    run_ui()
