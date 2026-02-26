from threading import Thread
import tkinter as tk
from RealtimeSTT import AudioToTextRecorder
from granola_voice import app

# Import your compiled langgraph app here
# from your_workflow_file import app 

class VoiceAgentGui:
    def __init__(self, langgraph_app):
        self.root = tk.Tk()
        self.root.title("Voice Agent")
        self.langgraph_app = langgraph_app  # Store the graph

        self.status_label = tk.Label(self.root, text="Status: Ready", fg="blue")
        self.status_label.pack(pady=10)

        # ... (Buttons as you defined them) ...
        self.start_btn = tk.Button(self.root, text="Start Recording", command=self.start)
        self.start_btn.pack()

        self.stop_btn = tk.Button(self.root, text="Stop recording", command=self.stop, state="disabled")
        self.stop_btn.pack()

        self.summarize_btn = tk.Button(self.root, text="Summarize", command=self.summarize)
        self.summarize_btn.pack()

        self.text_display = tk.Text(self.root, height=10, width=80)
        self.text_display.pack(pady=5)

        self.text_summary_display = tk.Text(self.root, height=10, width=80)
        self.text_summary_display.pack(pady=5)

        self.recording = False
        self.recorder = None
        self.text_buffer = ""

    def summarize(self):
        """Triggered by the button; starts the background thread."""
        if not self.text_buffer.strip():
            self.status_label.config(text="Status: No text to summarize", fg="orange")
            return

        self.status_label.config(text="Status: Summarizing...", fg="purple")
        self.summarize_btn.config(state="disabled")

        # Run LangGraph in a background thread to prevent UI freezing
        Thread(target=self._run_summarization_thread, daemon=True).start()

    def _run_summarization_thread(self):
        """The actual LangGraph call happens here."""
        try:
            # 1. Validate input
            text_to_process = self.text_buffer.strip()
            if not text_to_process:
                raise ValueError("Text buffer is empty. Record some audio first!")

            # 2. Prepare the initial state
            # Note: we MUST provide empty lists for Annotated keys
            initial_state = {
                "user_input": text_to_process,
                "summary": "",
                "required_actions": [],
                "response": []
            }
            
            # 3. Invoke the graph
            # This is where the actual logic happens
            result = self.langgraph_app.invoke(initial_state)
            
            # 4. Success! Update the UI
            self.root.after(0, self._update_ui_with_summary, result)

        except Exception as e:
            # Create a clean error message
            error_display = f"AI Error: {type(e).__name__} - {str(e)}"
            print(error_display) # Print to console so you can see the full Traceback
            
            # Use a proper function or a 'default argument' lambda to avoid NameError
            self.root.after(0, lambda m=error_display: self._handle_error_ui(m))

    def _handle_error_ui(self, message):
        """Safely updates UI on error"""
        self.status_label.config(text=message, fg="red")
        self.summarize_btn.config(state="normal")

    def _update_ui_with_summary(self, result):
        """Updates the GUI text box. Must be called on the main thread."""
        summary_text = result.get("summary", "No summary generated.")
        responses = "\n".join(result.get("response", []))
        
        full_display = f"SUMMARY:\n{summary_text}\n\nACTIONS TAKEN:\n{responses}"
        
        self.text_summary_display.delete("1.0", tk.END)
        self.text_summary_display.insert(tk.END, full_display)
        
        self.status_label.config(text="Status: Ready", fg="blue")
        self.summarize_btn.config(state="normal")

    # ... (Keep your existing on_speech, start, stop methods) ...
    def on_speech(self, text):
        self.text_display.insert(tk.END, f"User: {text}\n")
        self.text_display.see(tk.END)
        self.text_buffer += f"{text}\n"

    def start(self):
        self.recording = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="Listening...", fg="red")
        Thread(target=self._run_recorder, daemon=True).start()

    def _run_recorder(self):
        if not self.recorder:
            self.recorder = AudioToTextRecorder(model="tiny.en")
        while self.recording:
            self.recorder.text(self.on_speech)

    def stop(self):
        self.recording = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Stopped", fg="blue")

if __name__ == "__main__":
    # Ensure you pass your 'app' variable here
    gui = VoiceAgentGui(langgraph_app=app)
    gui.root.mainloop()
