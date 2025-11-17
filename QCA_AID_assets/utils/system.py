"""
System Utilities

Tkinter patches, input handling, and other system-level utilities.
"""

import threading
import sys
import time
from threading import Event
from typing import Optional


def patch_tkinter_for_threaded_exit() -> None:
    """
    Patcht die Tkinter Variable.__del__ Methode, um den RuntimeError beim Beenden zu vermeiden.
    
    This fixes issues when Tkinter GUI is running in a separate thread.
    """
    import tkinter
    
    # Store original __del__ method
    original_del = tkinter.Variable.__del__
    
    # Define new __del__ method that catches exceptions
    def safe_del(self):
        try:
            # Only call if _tk exists and is a valid Tkinter object
            if hasattr(self, '_tk') and self._tk:
                original_del(self)
        except (RuntimeError, TypeError, AttributeError):
            # Silently ignore these exceptions
            pass
    
    # Replace the original method
    tkinter.Variable.__del__ = safe_del
    print("Tkinter für sicheres Beenden gepatcht.")


def get_input_with_timeout(prompt: str, timeout: int = 30) -> str:
    """
    Fragt nach Benutzereingabe mit Timeout.
    
    Args:
        prompt: Anzuzeigender Text
        timeout: Timeout in Sekunden (default: 30)
        
    Returns:
        str: Benutzereingabe oder 'n' bei Timeout
    """
    # Platform-specific imports
    if sys.platform == 'win32':
        import msvcrt
    else:
        import select

    answer = {'value': None}
    stop_event = Event()
    
    def input_thread():
        try:
            # Show countdown
            remaining_time = timeout
            while remaining_time > 0 and not stop_event.is_set():
                sys.stdout.write(f'\r{prompt} ({remaining_time}s): ')
                sys.stdout.flush()
                
                # Platform-specific input check
                if sys.platform == 'win32':
                    if msvcrt.kbhit():
                        answer['value'] = msvcrt.getche().decode().strip().lower()
                        sys.stdout.write('\n')
                        stop_event.set()
                        return
                else:
                    if select.select([sys.stdin], [], [], 1)[0]:
                        answer['value'] = sys.stdin.readline().strip().lower()
                        stop_event.set()
                        return
                
                time.sleep(1)
                remaining_time -= 1
            
            # On timeout
            if not stop_event.is_set():
                sys.stdout.write('\n')
                sys.stdout.flush()
                
        except (KeyboardInterrupt, EOFError):
            stop_event.set()
    
    # Start input thread
    thread = threading.Thread(target=input_thread)
    thread.daemon = True
    thread.start()
    
    # Wait for response or timeout
    thread.join(timeout)
    stop_event.set()
    
    if answer['value'] is None:
        print(f"\nKeine Eingabe innerhalb von {timeout} Sekunden - verwende 'n'")
        return 'n'
        
    return answer['value']


__all__ = [
    'patch_tkinter_for_threaded_exit',
    'get_input_with_timeout',
]
