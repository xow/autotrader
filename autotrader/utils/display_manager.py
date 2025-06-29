import os
import sys
from colorama import Fore, Style, init

init(autoreset=True)

class DisplayManager:
    """
    Manages terminal output for the AutoTrader bot, providing clean,
    live-updating, and aesthetically pleasing information display.
    """
    
    def __init__(self, total_iterations: int = 0):
        self.total_iterations = total_iterations
        self.start_time = None
        self._last_display_lines = 0 # To keep track of how many lines were last printed

    def _clear_lines(self, num_lines: int):
        """Clears the specified number of lines and moves cursor up."""
        for _ in range(num_lines):
            sys.stdout.write("\033[K") # Clear line from cursor to end
            sys.stdout.write("\033[1A") # Move cursor up one line
        sys.stdout.write("\r") # Move cursor to beginning of the line
        sys.stdout.flush()

    def _format_progress_bar(self, current_value: int, total_value: int, bar_length: int = 40) -> str:
        """Generates a text-based progress bar."""
        if total_value == 0:
            return "[" + " " * bar_length + "]"
        
        progress = (current_value / total_value)
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + ' ' * (bar_length - filled_length) # Changed '-' to ' ' for a cleaner look
        return f"[{bar}]"

    def display_status(self, iteration: int, status: str, price: float, signal_type: str,
                       confidence: float, balance: float, position_size: float,
                       rsi: float, training_samples: int = 0, total_training_samples: int = 0):
        """
        Displays the main trading bot status line with live updates.
        """
        self._clear_lines(self._last_display_lines) # Clear previous output

        # Determine status indicator color
        status_color = Fore.GREEN if status == "ACTIVE" else Fore.YELLOW
        status_symbol = "●" if status == "ACTIVE" else "○"

        # Format signal with color
        signal_color = Fore.GREEN if signal_type == "BUY" else (
                       Fore.RED if signal_type == "SELL" else Fore.YELLOW)
        
        iteration_str = f"{iteration}/∞" if self.total_iterations == 0 else f"{iteration}/{self.total_iterations}"

        header_part = (
            f"{Fore.CYAN}{Style.BRIGHT}AutoTrader Bot{Style.RESET_ALL} | "
            f"Iteration {iteration_str} | "
            f"Status: {status_color}{status_symbol}{status}{Style.RESET_ALL}"
        )
        
        main_info_part = (
            f"Price: {Fore.WHITE}${price:,.2f}{Style.RESET_ALL} | "
            f"Signal: {signal_color}{signal_type}{Style.RESET_ALL} ({confidence:.0%}) | "
            f"RSI: {Fore.WHITE}{rsi:.1f}{Style.RESET_ALL} | "
            f"Balance: {Fore.GREEN}${balance:,.2f}{Style.RESET_ALL} | "
            f"BTC: {Fore.YELLOW}{position_size:.4f}{Style.RESET_ALL}"
        )
        
        terminal_width = os.get_terminal_size().columns

        first_line = header_part
        second_line = main_info_part
        
        # Training progress bar
        if total_training_samples > 0:
            training_bar = self._format_progress_bar(training_samples, total_training_samples)
            training_status_line = f"{training_bar} Training: {training_samples} samples"
        else:
            training_status_line = ""

        # Print each line
        sys.stdout.write(first_line.ljust(terminal_width) + "\n")
        sys.stdout.write(second_line.ljust(terminal_width) + "\n")
        
        current_lines_displayed = 2
        if training_status_line:
            sys.stdout.write(training_status_line.ljust(terminal_width) + "\n")
            current_lines_displayed += 1

        sys.stdout.flush()
        self._last_display_lines = current_lines_displayed

        # Now, move the cursor back up to the start of the first line for the next update
        sys.stdout.write(f"\033[{current_lines_displayed}A") # Move cursor up
        sys.stdout.write("\r") # Move cursor to beginning of the line
        sys.stdout.flush()

    def log_message(self, message: str, level: str = "info"):
        """
        Logs a message to the console without interfering with the status line.
        It clears the live status, prints the message, and then relies on the main loop
        to re-display the status on its next iteration.
        """
        # Clear the live status lines before printing a log message
        self._clear_lines(self._last_display_lines) 
        
        if level == "info":
            sys.stdout.write(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {message}\n")
        elif level == "warning":
            sys.stdout.write(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {message}\n")
        elif level == "error":
            sys.stdout.write(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {message}\n")
        elif level == "debug":
            sys.stdout.write(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} {message}\n")
        else:
            sys.stdout.write(f"[LOG] {message}\n")
        sys.stdout.flush()

    def print_initialization_message(self):
        """Prints a clean, concise initialization message."""
        sys.stdout.write(f"{Fore.CYAN}{Style.BRIGHT}AutoTrader Bot Initializing...{Style.RESET_ALL}\n")
        sys.stdout.flush()

    def print_shutdown_message(self):
        """Prints a clean shutdown message."""
        self._clear_line()
        sys.stdout.write(f"{Fore.CYAN}{Style.BRIGHT}AutoTrader Bot Shutting Down. Goodbye!{Style.RESET_ALL}\n")
        sys.stdout.flush()

    def print_training_start(self):
        """Prints a message when training starts."""
        self.log_message("Initiating model training...", level="info")

    def print_training_complete(self, loss: float, mae: float):
        """Prints a message when training is complete."""
        self.log_message(f"Model training complete. Loss: {loss:.4f}, MAE: {mae:.4f}", level="info")

    def print_data_save(self, num_samples: int):
        """Prints a message when training data is saved."""
        self.log_message(f"Training data saved: {num_samples} samples.", level="info")

    def print_model_save(self, filename: str):
        """Prints a message when the model is saved."""
        self.log_message(f"Model saved to: {filename}", level="info")

    def print_scalers_save(self):
        """Prints a message when scalers are saved."""
        self.log_message("Scalers saved successfully.", level="info")

    def print_state_save(self):
        """Prints a message when the state is saved."""
        self.log_message("Trader state saved successfully.", level="info")

    def print_market_data_fetch_fail(self):
        """Prints a message when market data fetching fails."""
        self.log_message("Failed to collect market data. Skipping iteration.", level="warning")

    def print_no_training_data(self):
        """Prints a message when no training data is available for prediction."""
        self.log_message("No training data available for prediction. Skipping trade signal generation.", level="warning")

    def print_insufficient_training_data_for_scaler(self, needed: int, got: int):
        """Prints a message for insufficient data for scaler fitting."""
        self.log_message(f"Insufficient training data for scaler fitting. Needed: {needed}, Got: {got}.", level="warning")

    def print_model_training_failed(self):
        """Prints a message when model training fails or is skipped."""
        self.log_message("Model training failed or skipped due to insufficient data.", level="warning")

    def print_trade_executed(self, signal_type: str, amount: float, price: float, new_balance: float, position: float):
        """Prints details when a trade is executed."""
        if signal_type == "BUY":
            self.log_message(f"{Fore.GREEN}BUY executed:{Style.RESET_ALL} Amount: {amount:.4f}, Price: ${price:,.2f}, New Balance: ${new_balance:,.2f}, Position: {position:.4f} BTC", level="info")
        elif signal_type == "SELL":
            self.log_message(f"{Fore.RED}SELL executed:{Style.RESET_ALL} Amount: {amount:.4f}, Price: ${price:,.2f}, New Balance: ${new_balance:,.2f}, Position: {position:.4f} BTC", level="info")

    def print_insufficient_funds(self, action: str, needed: float, have: float):
        """Prints a warning for insufficient funds/position."""
        self.log_message(f"Insufficient {'balance' if action == 'BUY' else 'position'} to {action}. Needed: ${needed:,.2f}, Have: ${have:,.2f}", level="warning")

    def print_hold_signal(self, confidence: float, rsi: float):
        """Prints a message for a HOLD signal."""
        self.log_message(f"HOLD signal. Confidence: {confidence:.2f}, RSI: {rsi:.2f}", level="info")

    def print_pnl(self, pnl: float, pnl_pct: float):
        """Prints current PnL."""
        color = Fore.GREEN if pnl >= 0 else Fore.RED
        self.log_message(f"Current PnL: {color}${pnl:,.2f}{Style.RESET_ALL} ({color}{pnl_pct:.2f}%{Style.RESET_ALL})", level="info")

    def print_rsi_override(self, signal_type: str, rsi: float):
        """Prints a message when RSI overrides a trade signal."""
        self.log_message(f"RSI is {'overbought' if signal_type == 'BUY' else 'oversold'}, overriding {signal_type} signal to HOLD. RSI: {rsi:.2f}", level="info")
