#!/usr/bin/env python3
import subprocess, sys, os, argparse, json, threading, queue, time, signal
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from enum import Enum

class ModelType(Enum):
    GPT_OSS = "gpt-oss"
    DEEPSEEK = "deepseek"

class ModelConfig:
    CONFIGS = {
        "OpenAI-20B-NEO-CODE2-Plus-Uncensored-IQ4_NL.gguf": {
            "type": ModelType.GPT_OSS, "name": "OpenAI GPT-OSS 20B (NEO-CODE2)",
            "default_temp": "0.7", "default_top_k": "40", "default_top_p": "0.95",
            "default_max_tokens": "500", "supports_harmony": True, "supports_simple": True
        },
        "deepseek-coder-6.7b-instruct.Q3_K_M.gguf": {
            "type": ModelType.DEEPSEEK, "name": "DeepSeek Coder 6.7B",
            "default_temp": "0.1", "default_top_k": "40", "default_top_p": "0.95",
            "default_max_tokens": "500", "supports_harmony": False, "supports_simple": True
        }
    }

class PromptFormatter:
    @staticmethod
    def format_gpt_oss_harmony(instruction, reasoning="medium", system_prompt=""):
        prompt = f"""<|start|>system<|message|>You are a helpful AI assistant.
Knowledge cutoff: 2024-06
Current date: 2025-09-11
Reasoning: {reasoning}
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"""
        if system_prompt:
            prompt += f"\n<|start|>developer<|message|># Instructions\n{system_prompt}<|end|>"
        return prompt + f"\n<|start|>user<|message|>{instruction}<|end|>\n<|start|>assistant"
    
    @staticmethod
    def format_gpt_oss_simple(instruction): return instruction
    
    @staticmethod
    def format_deepseek(instruction, use_template=True):
        return f"""You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science.
### Instruction:
{instruction}
### Response:""" if use_template else instruction

class LLMGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Local LLM")
        self.root.geometry("1200x900")
        ttk.Style().theme_use('clam')
        
        self.llama_cli = self.find_llama_cli()
        self.selected_model = tk.StringVar()
        self.prompt_format = tk.StringVar(value="auto")
        self.reasoning_level = tk.StringVar(value="medium")
        self.output_mode = tk.StringVar(value="gui")
        self.process = None
        self.output_queue = queue.Queue()
        self.generation_thread = None
        self.is_generating = False
        self.available_models = self.detect_models()
        
        self.create_widgets()
        if self.available_models:
            self.selected_model.set(self.available_models[0])
            self.on_model_change()
        self.process_queue()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def find_llama_cli(self):
        paths = ["llama-cli.exe", "./llama-cli.exe", "llama-cli", "./llama-cli", "../llama-cli.exe", "../llama-cli"]
        for path in paths:
            if os.path.isfile(path): return os.path.abspath(path)
        messagebox.showwarning("llama-cli not found", "Could not find llama-cli executable.\nPlease ensure llama-cli.exe is in the current directory.")
        return "llama-cli.exe"
    
    def on_closing(self):
        if self.is_generating and not messagebox.askokcancel("Quit", "Generation is in progress. Force quit?"): return
        self.force_shutdown() if self.is_generating else self.root.destroy()
    
    def detect_models(self):
        return [m for m in ModelConfig.CONFIGS.keys() if Path(m).exists()]
    
    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text='Input')
        self.create_input_widgets(input_frame)
        
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text='Output')
        self.create_output_widgets(output_frame)
        
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, relief=tk.SUNKEN).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
    
    def create_input_widgets(self, parent):
        main = ttk.Frame(parent, padding="10")
        main.pack(fill='both', expand=True)
        
        # Model Selection
        mf = ttk.LabelFrame(main, text="Model Selection", padding="10")
        mf.pack(fill='x', pady=5)
        ttk.Label(mf, text="Select Model:").grid(row=0, column=0, sticky=tk.W, padx=5)
        
        if self.available_models:
            model_combo = ttk.Combobox(mf, textvariable=self.selected_model, values=self.available_models, state="readonly", width=50)
            model_combo.grid(row=0, column=1, padx=5)
            model_combo.bind('<<ComboboxSelected>>', lambda e: self.on_model_change())
        else:
            ttk.Label(mf, text="No models found!", foreground="red").grid(row=0, column=1, padx=5)
        
        self.model_info_label = ttk.Label(mf, text="", foreground="blue")
        self.model_info_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        ef = ttk.Frame(mf)
        ef.grid(row=2, column=0, columnspan=2, pady=5)
        self.exec_label = ttk.Label(ef, text=f"Executable: {os.path.basename(self.llama_cli)}", foreground="gray")
        self.exec_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(ef, text="Browse...", command=self.browse_executable).pack(side=tk.LEFT, padx=5)
        
        # Output Mode
        omf = ttk.LabelFrame(main, text="Output Mode", padding="10")
        omf.pack(fill='x', pady=5)
        ttk.Radiobutton(omf, text="Show in GUI", variable=self.output_mode, value="gui").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(omf, text="Show in Terminal (Recommended for large models)", variable=self.output_mode, value="terminal").pack(side=tk.LEFT, padx=10)
        
        # Prompt Format
        ff = ttk.LabelFrame(main, text="Prompt Format", padding="10")
        ff.pack(fill='x', pady=5)
        ttk.Label(ff, text="Format Type:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.format_combo = ttk.Combobox(ff, textvariable=self.prompt_format, state="readonly", width=30)
        self.format_combo.grid(row=0, column=1, padx=5)
        ttk.Label(ff, text="Reasoning Level (GPT-OSS only):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Combobox(ff, textvariable=self.reasoning_level, values=["low", "medium", "high"], state="readonly", width=30).grid(row=1, column=1, padx=5, pady=5)
        
        # Parameters
        pf = ttk.LabelFrame(main, text="Generation Parameters", padding="10")
        pf.pack(fill='x', pady=5)
        self.params = {}
        for i, (l, k, d) in enumerate([("Max Tokens:", "max_tokens", "500"), ("Temperature:", "temperature", "0.7"),
                                        ("Top-K:", "top_k", "40"), ("Top-P:", "top_p", "0.95"),
                                        ("Threads:", "threads", "8"), ("Context Size:", "context", "4096")]):
            r, c = i // 3, (i % 3) * 2
            ttk.Label(pf, text=l).grid(row=r, column=c, sticky=tk.W, padx=5, pady=2)
            e = ttk.Entry(pf, width=10)
            e.insert(0, d)
            e.grid(row=r, column=c+1, padx=5, pady=2)
            self.params[k] = e
        
        # Prompt Input
        pif = ttk.LabelFrame(main, text="Prompt Input", padding="10")
        pif.pack(fill='both', expand=True, pady=5)
        bf = ttk.Frame(pif)
        bf.pack(fill='x', pady=5)
        ttk.Button(bf, text="Load from File", command=self.load_prompt_from_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Clear", command=self.clear_prompt).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Load Example", command=self.load_example).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(pif, text="System Prompt (optional):").pack(anchor='w')
        self.system_prompt = scrolledtext.ScrolledText(pif, height=3, wrap=tk.WORD)
        self.system_prompt.pack(fill='x', pady=5)
        ttk.Label(pif, text="User Prompt:").pack(anchor='w')
        self.prompt_text = scrolledtext.ScrolledText(pif, height=10, wrap=tk.WORD)
        self.prompt_text.pack(fill='both', expand=True, pady=5)
        
        # Action Buttons
        af = ttk.Frame(main)
        af.pack(pady=10)
        self.generate_btn = ttk.Button(af, text="Generate", command=self.generate, state=tk.NORMAL if self.available_models else tk.DISABLED)
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(af, text="Stop Generation", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.force_shutdown_btn = ttk.Button(af, text="FORCE SHUTDOWN", command=self.force_shutdown, state=tk.DISABLED)
        self.force_shutdown_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(af, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(af, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
    
    def create_output_widgets(self, parent):
        oc = ttk.Frame(parent, padding="10")
        oc.pack(fill='both', expand=True)
        
        cf = ttk.Frame(oc)
        cf.pack(fill='x', pady=5)
        ttk.Button(cf, text="Clear Output", command=self.clear_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(cf, text="Copy Output", command=self.copy_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(cf, text="Save Output", command=self.save_output).pack(side=tk.LEFT, padx=5)
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(cf, text="Auto-scroll", variable=self.auto_scroll_var).pack(side=tk.LEFT, padx=20)
        ttk.Label(cf, text="⚠ If output freezes, use Terminal mode instead", foreground="orange").pack(side=tk.RIGHT, padx=10)
        
        ttk.Label(oc, text="Model Output:").pack(anchor='w')
        of = ttk.Frame(oc)
        of.pack(fill='both', expand=True, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(of, height=30, wrap=tk.WORD, bg='black', fg='white',
                                                     font=('Consolas', 10) if sys.platform == 'win32' else ('Courier', 10),
                                                     insertbackground='white')
        self.output_text.pack(fill='both', expand=True)
        for tag, color in [('prompt', 'cyan'), ('info', 'yellow'), ('error', 'red'), ('success', 'green'), ('warning', 'orange')]:
            self.output_text.tag_config(tag, foreground=color)
    
    def browse_executable(self):
        fp = filedialog.askopenfilename(title="Select llama-cli executable",
                                        filetypes=[("Executable files", "*.exe" if sys.platform == "win32" else "*"), ("All files", "*.*")])
        if fp and os.path.isfile(fp):
            self.llama_cli = os.path.abspath(fp)
            self.exec_label.config(text=f"Executable: {os.path.basename(self.llama_cli)}")
            self.status_var.set(f"Executable set to: {os.path.basename(self.llama_cli)}")
    
    def on_model_change(self):
        model = self.selected_model.get()
        if model not in ModelConfig.CONFIGS: return
        cfg = ModelConfig.CONFIGS[model]
        self.model_info_label.config(text=f"Model: {cfg['name']}")
        
        formats = []
        if cfg["supports_simple"]: formats.append("simple")
        if cfg["supports_harmony"]: formats.append("harmony")
        formats.append("auto")
        self.format_combo['values'] = formats
        self.prompt_format.set("auto")
        
        for p in ["max_tokens", "temperature", "top_k", "top_p"]:
            self.params[p].delete(0, tk.END)
            self.params[p].insert(0, cfg[f"default_{p.replace('_tokens', '_tokens').replace('temperature', 'temp')}"])
    
    def load_prompt_from_file(self):
        fp = filedialog.askopenfilename(title="Select prompt file", filetypes=[("Text files", "*.txt"), ("Markdown files", "*.md"), ("All files", "*.*")])
        if fp:
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    self.prompt_text.delete(1.0, tk.END)
                    self.prompt_text.insert(1.0, f.read())
                    self.status_var.set(f"Loaded: {Path(fp).name}")
            except Exception as e: messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def clear_prompt(self):
        self.prompt_text.delete(1.0, tk.END)
        self.system_prompt.delete(1.0, tk.END)
        self.status_var.set("Cleared")
    
    def clear_output(self): self.output_text.delete(1.0, tk.END)
    
    def copy_output(self):
        o = self.output_text.get(1.0, tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(o)
        self.status_var.set("Output copied to clipboard")
    
    def save_output(self):
        fp = filedialog.asksaveasfilename(title="Save output", defaultextension=".txt",
                                          filetypes=[("Text files", "*.txt"), ("Markdown files", "*.md"), ("All files", "*.*")])
        if fp:
            try:
                with open(fp, 'w', encoding='utf-8') as f:
                    f.write(self.output_text.get(1.0, tk.END))
                    self.status_var.set(f"Output saved to: {Path(fp).name}")
            except Exception as e: messagebox.showerror("Error", f"Failed to save: {e}")
    
    def load_example(self):
        model = self.selected_model.get()
        if model not in ModelConfig.CONFIGS: return
        cfg = ModelConfig.CONFIGS[model]
        self.prompt_text.delete(1.0, tk.END)
        self.system_prompt.delete(1.0, tk.END)
        if cfg["type"] == ModelType.DEEPSEEK:
            self.prompt_text.insert(1.0, "Write a Python function to calculate the Fibonacci sequence recursively with memoization.")
            self.system_prompt.insert(1.0, "You are an expert Python programmer. Write clean, efficient, and well-documented code.")
        else:
            self.prompt_text.insert(1.0, "Explain the concept of quantum entanglement in simple terms.")
            self.system_prompt.insert(1.0, "You are a helpful AI assistant that explains complex topics clearly.")
    
    def format_prompt(self, instruction, system_prompt=""):
        model = self.selected_model.get()
        if model not in ModelConfig.CONFIGS: return instruction
        
        cfg = ModelConfig.CONFIGS[model]
        mt = cfg["type"]
        ft = self.prompt_format.get()
        
        if ft == "auto":
            if mt == ModelType.DEEPSEEK: return PromptFormatter.format_deepseek(instruction, True)
            elif mt == ModelType.GPT_OSS:
                return PromptFormatter.format_gpt_oss_simple(instruction) if "uncensored" in model.lower() \
                       else PromptFormatter.format_gpt_oss_harmony(instruction, self.reasoning_level.get(), system_prompt)
        elif ft == "harmony": return PromptFormatter.format_gpt_oss_harmony(instruction, self.reasoning_level.get(), system_prompt)
        elif ft == "simple": return PromptFormatter.format_deepseek(instruction, False) if mt == ModelType.DEEPSEEK else PromptFormatter.format_gpt_oss_simple(instruction)
        return instruction
    
    def generate(self):
        if not self.available_models:
            messagebox.showerror("Error", "No models available!")
            return
        if self.is_generating:
            messagebox.showwarning("Warning", "Generation already in progress!")
            return
        
        instruction = self.prompt_text.get(1.0, tk.END).strip()
        if not instruction:
            messagebox.showwarning("Warning", "Please enter a prompt!")
            return
        
        system_prompt = self.system_prompt.get(1.0, tk.END).strip()
        formatted_prompt = self.format_prompt(instruction, system_prompt)
        model_path = self.selected_model.get()
        
        cmd = [self.llama_cli, '-m', model_path, '-p', formatted_prompt,
               '-n', self.params["max_tokens"].get(), '-t', self.params["threads"].get(),
               '-c', self.params["context"].get(), '--temp', self.params["temperature"].get(),
               '--top-k', self.params["top_k"].get(), '--top-p', self.params["top_p"].get(),
               '--repeat-penalty', '1.1']
        
        self.run_in_terminal(cmd, formatted_prompt, model_path) if self.output_mode.get() == "terminal" \
        else self.run_in_gui(cmd, formatted_prompt, model_path)
    
    def run_in_terminal(self, cmd, formatted_prompt, model_path):
        print("\n" + "="*60)
        print(f"Model: {ModelConfig.CONFIGS[model_path]['name']}")
        print(f"Format: {self.prompt_format.get()}\nExecutable: {self.llama_cli}")
        print("="*60 + "\nFormatted Prompt:\n" + "-"*40)
        print(formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt)
        print("-"*40 + "\nModel Output:\n" + "-"*40)
        
        self.status_var.set("Generating in terminal...")
        try:
            self.process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE) if sys.platform == "win32" else subprocess.Popen(cmd)
            self.is_generating = True
            self.generate_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.force_shutdown_btn.config(state=tk.NORMAL)
            threading.Thread(target=self.monitor_terminal_process, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run model: {e}")
            self.reset_ui()
    
    def monitor_terminal_process(self):
        if self.process:
            self.process.wait()
            self.output_queue.put(('terminal_done', None))
    
    def run_in_gui(self, cmd, formatted_prompt, model_path):
        self.clear_output()
        self.notebook.select(1)
        
        sep = "="*60
        self.output_text.insert(tk.END, f"{sep}\nModel: {ModelConfig.CONFIGS[model_path]['name']}\n", 'info')
        self.output_text.insert(tk.END, f"Format: {self.prompt_format.get()}\nExecutable: {self.llama_cli}\n{sep}\n", 'info')
        self.output_text.insert(tk.END, f"Formatted Prompt:\n{'-'*40}\n", 'info')
        self.output_text.insert(tk.END, (formatted_prompt[:500] + "..." if len(formatted_prompt) > 500 else formatted_prompt) + "\n", 'prompt')
        self.output_text.insert(tk.END, f"{'-'*40}\nModel Output:\n{'-'*40}\n\n", 'info')
        self.output_text.insert(tk.END, "[Starting generation... If no output appears within 30 seconds, use Terminal mode]\n\n", 'warning')
        self.output_text.see(tk.END)
        
        self.is_generating = True
        self.generate_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.force_shutdown_btn.config(state=tk.NORMAL)
        self.status_var.set("Generating...")
        
        self.generation_thread = threading.Thread(target=self.run_generation_gui, args=(cmd,), daemon=True)
        self.generation_thread.start()
    
    def run_generation_gui(self, cmd):
        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           universal_newlines=True, bufsize=0, encoding='utf-8', errors='replace')
            buffer = ""
            while True:
                if self.process.poll() is not None: break
                try:
                    char = self.process.stdout.read(1)
                    if not char: break
                    buffer += char
                    if char == '\n' or len(buffer) > 100:
                        self.output_queue.put(('output', buffer))
                        buffer = ""
                except Exception as e:
                    self.output_queue.put(('error', f"Read error: {str(e)}\n"))
                    break
            
            if buffer: self.output_queue.put(('output', buffer))
            if self.process:
                remaining = self.process.stdout.read()
                if remaining: self.output_queue.put(('output', remaining))
            self.output_queue.put(('done', None))
        except Exception as e:
            self.output_queue.put(('error', f"Error running model: {str(e)}\n"))
            self.output_queue.put(('done', None))
    
    def process_queue(self):
        try:
            while not self.output_queue.empty():
                msg_type, content = self.output_queue.get_nowait()
                if msg_type == 'output':
                    self.output_text.insert(tk.END, content)
                    if self.auto_scroll_var.get():
                        self.output_text.see(tk.END)
                        self.output_text.update_idletasks()
                elif msg_type == 'error':
                    self.output_text.insert(tk.END, content, 'error')
                    if self.auto_scroll_var.get(): self.output_text.see(tk.END)
                elif msg_type in ['done', 'terminal_done']:
                    if msg_type == 'done':
                        self.output_text.insert(tk.END, f"\n{'='*60}\nGeneration complete!\n{'='*60}\n", 'success')
                    self.reset_ui()
                    self.status_var.set("Generation complete")
        except: pass
        self.root.after(50, self.process_queue)
    
    def reset_ui(self):
        self.is_generating = False
        self.process = None
        self.generate_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.force_shutdown_btn.config(state=tk.DISABLED)
    
    def stop_generation(self):
        if self.process:
            try:
                self.process.terminate() if sys.platform == "win32" else self.process.send_signal(signal.SIGINT)
                self.output_text.insert(tk.END, "\n\n[Generation stopped by user]\n", 'warning')
                self.status_var.set("Generation stopped")
                time.sleep(0.5)
                if self.process.poll() is None: self.process.kill()
                self.reset_ui()
            except: self.force_shutdown()
    
    def force_shutdown(self):
        try:
            if self.process:
                try: self.process.kill()
                except: pass
                if sys.platform == "win32":
                    try: os.system(f"taskkill /F /PID {self.process.pid}")
                    except: pass
            self.output_text.insert(tk.END, "\n\n[FORCED SHUTDOWN]\n", 'error')
            self.reset_ui()
            self.status_var.set("Force shutdown complete")
            if messagebox.askyesno("Shutdown Complete", "Force shutdown complete. Exit application?"):
                self.root.destroy()
                sys.exit(0)
        except Exception as e:
            messagebox.showerror("Critical Error", f"Failed to force shutdown: {e}\nYou may need to restart the application.")
            self.root.destroy()
            sys.exit(1)
    
    def save_config(self):
        config = {"model": self.selected_model.get(), "format": self.prompt_format.get(),
                 "reasoning": self.reasoning_level.get(), "output_mode": self.output_mode.get(),
                 "parameters": {k: e.get() for k, e in self.params.items()},
                 "system_prompt": self.system_prompt.get(1.0, tk.END).strip(),
                 "user_prompt": self.prompt_text.get(1.0, tk.END).strip()}
        
        fp = filedialog.asksaveasfilename(title="Save configuration", defaultextension=".json",
                                          filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if fp:
            try:
                with open(fp, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                    self.status_var.set(f"Saved config: {Path(fp).name}")
            except Exception as e: messagebox.showerror("Error", f"Failed to save config: {e}")
    
    def load_config(self):
        fp = filedialog.askopenfilename(title="Load configuration", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not fp: return
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if "model" in cfg and cfg["model"] in self.available_models:
                self.selected_model.set(cfg["model"])
                self.on_model_change()
            for k in ["format", "reasoning", "output_mode"]:
                if k in cfg: getattr(self, f"prompt_{k}" if k == "format" else f"{k}_level" if k == "reasoning" else f"{k}").set(cfg[k])
            if "parameters" in cfg:
                for k, v in cfg["parameters"].items():
                    if k in self.params:
                        self.params[k].delete(0, tk.END)
                        self.params[k].insert(0, v)
            for k, w in [("system_prompt", self.system_prompt), ("user_prompt", self.prompt_text)]:
                if k in cfg:
                    w.delete(1.0, tk.END)
                    w.insert(1.0, cfg[k])
            self.status_var.set(f"Loaded config: {Path(fp).name}")
        except Exception as e: messagebox.showerror("Error", f"Failed to load config: {e}")

def main():
    if len(sys.argv) > 1:
        print("Command line mode not implemented in GUI version.\nRun without arguments to launch GUI.")
        sys.exit(1)
    root = tk.Tk()
    app = LLMGui(root)
    root.mainloop()

if __name__ == "__main__":
    main()