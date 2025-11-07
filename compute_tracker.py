import torch
import sys
import pickle
import json
from collections import defaultdict

class ComputeTracker:
    """Minimal tracker for GPU memory and communication costs"""
    
    def __init__(self, device, role="client"):
        self.device = device
        self.role = role
        self.is_cuda = torch.cuda.is_available() and device.type == 'cuda'
        
        # Phases: 'train' and 'eval'
        self.phase = 'train'
        # GPU memory tracking (overall)
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        # GPU memory tracking per-phase (use instantaneous current to form per-phase peak)
        self.phase_peak_memory_mb = { 'train': 0.0, 'eval': 0.0 }
        
        # Communication tracking (overall)
        self.comm_rounds = 0
        self.bytes_sent = 0
        self.bytes_received = 0
        # Communication tracking per-phase
        self.phase_comm_rounds = { 'train': 0, 'eval': 0 }
        self.phase_bytes_sent = { 'train': 0, 'eval': 0 }
        self.phase_bytes_received = { 'train': 0, 'eval': 0 }
        self.forward_passes = 0
        self.backward_passes = 0
        
        # Per-step tracking
        self.step_stats = []

    def set_phase(self, phase: str):
        """Switch accounting to 'train' or 'eval'."""
        if phase not in ('train','eval'):
            return
        # Reset CUDA peak counter when switching phases so per-phase peaks are accurate
        if phase != self.phase and self.is_cuda:
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass
        self.phase = phase
        
    def update_memory(self):
        """Update GPU memory stats"""
        if self.is_cuda:
            current = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
            peak = torch.cuda.max_memory_allocated(self.device) / (1024**2)  # MB
            self.current_memory_mb = current
            self.peak_memory_mb = max(self.peak_memory_mb, peak)
            # Update per-phase peak as the max instantaneous current observed while in this phase
            try:
                # Use CUDA peak to avoid missing short-lived spikes within the phase
                self.phase_peak_memory_mb[self.phase] = max(self.phase_peak_memory_mb.get(self.phase, 0.0), peak)
            except Exception:
                pass
            return current, peak
        return 0.0, 0.0
    
    def log_memory(self, label=""):
        """Log current memory state"""
        if self.is_cuda:
            current, peak = self.update_memory()
            print(f"[{self.role}] GPU Memory {label}: Current={current:.2f}MB, Peak={peak:.2f}MB")
            return current, peak
        return 0.0, 0.0
    
    def track_send(self, data):
        """Track data sent over network"""
        size_bytes = self._get_size(data)
        self.bytes_sent += size_bytes
        self.comm_rounds += 1
        # Phase accounting
        try:
            self.phase_bytes_sent[self.phase] = self.phase_bytes_sent.get(self.phase, 0) + size_bytes
            self.phase_comm_rounds[self.phase] = self.phase_comm_rounds.get(self.phase, 0) + 1
        except Exception:
            pass
        return size_bytes
    
    def track_receive(self, data):
        """Track data received over network"""
        size_bytes = self._get_size(data)
        self.bytes_received += size_bytes
        try:
            self.phase_bytes_received[self.phase] = self.phase_bytes_received.get(self.phase, 0) + size_bytes
        except Exception:
            pass
        return size_bytes
    
    def _get_size(self, obj):
        """Estimate size of object in bytes"""
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, dict):
            total = 0
            for k, v in obj.items():
                total += self._get_size(v)
            return total
        elif isinstance(obj, (list, tuple)):
            return sum(self._get_size(item) for item in obj)
        else:
            try:
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            except:
                return sys.getsizeof(obj)
    
    def log_step(self, step, extra_info=None):
        """Log stats for current step"""
        current_mem, peak_mem = self.update_memory()
        
        step_data = {
            'step': step,
            'current_memory_mb': current_mem,
            'peak_memory_mb': peak_mem,
            'comm_rounds': self.comm_rounds,
            'bytes_sent_mb': self.bytes_sent / (1024**2),
            'bytes_received_mb': self.bytes_received / (1024**2),
        }
        
        if extra_info:
            step_data.update(extra_info)
        
        self.step_stats.append(step_data)
    
    def print_summary(self):
        """Print final summary statistics"""
        print(f"\n{'='*60}")
        print(f"[{self.role.upper()}] COMPUTE & COMMUNICATION SUMMARY")
        print(f"{'='*60}")
        
        # GPU Stats
        if self.is_cuda:
            print(f"GPU Memory:")
            print(f"  Peak Memory: {self.peak_memory_mb:.2f} MB")
            print(f"  Current Memory: {self.current_memory_mb:.2f} MB")
            # Per-phase peaks
            print(f"  Train Peak: {self.phase_peak_memory_mb.get('train',0.0):.2f} MB")
            print(f"  Eval  Peak: {self.phase_peak_memory_mb.get('eval',0.0):.2f} MB")
        else:
            print(f"GPU: Not available (CPU mode)")
        
        # Communication Stats
        total_data_mb = (self.bytes_sent + self.bytes_received) / (1024**2)
        print(f"\nCommunication:")
        print(f"  Total Rounds: {self.comm_rounds}")
        print(f"  Data Sent: {self.bytes_sent / (1024**2):.2f} MB")
        print(f"  Data Received: {self.bytes_received / (1024**2):.2f} MB")
        print(f"  Total Data: {total_data_mb:.2f} MB")
        print(f"  Avg per Round: {total_data_mb / max(1, self.comm_rounds):.2f} MB")

        # Per-phase communication
        print(f"\nCommunication by Phase:")
        for ph in ('train','eval'):
            rounds = int(self.phase_comm_rounds.get(ph, 0))
            sent_mb = self.phase_bytes_sent.get(ph, 0) / (1024**2)
            recv_mb = self.phase_bytes_received.get(ph, 0) / (1024**2)
            tot_mb  = sent_mb + recv_mb
            print(f"  {ph.title()}: rounds={rounds}, sent={sent_mb:.2f} MB, recv={recv_mb:.2f} MB, total={tot_mb:.2f} MB")
        
        print(f"\nTraining Passes:")
        print(f"  Forward Passes: {self.forward_passes}")
        print(f"  Backward Passes: {self.backward_passes}")
        
        print(f"{'='*60}\n")
        
        return {
            'peak_memory_mb': self.peak_memory_mb,
            'total_data_mb': total_data_mb,
            'comm_rounds': self.comm_rounds,
            'forward_passes': self.forward_passes,
            'backward_passes': self.backward_passes,
            'phase': {
                'train': {
                    'peak_memory_mb': self.phase_peak_memory_mb.get('train', 0.0),
                    'bytes_sent': self.phase_bytes_sent.get('train', 0),
                    'bytes_received': self.phase_bytes_received.get('train', 0),
                    'comm_rounds': self.phase_comm_rounds.get('train', 0),
                },
                'eval': {
                    'peak_memory_mb': self.phase_peak_memory_mb.get('eval', 0.0),
                    'bytes_sent': self.phase_bytes_sent.get('eval', 0),
                    'bytes_received': self.phase_bytes_received.get('eval', 0),
                    'comm_rounds': self.phase_comm_rounds.get('eval', 0),
                }
            }
        }
    
    def save_stats(self, filepath):
        """Save detailed stats to JSON file"""
        stats = {
            'role': self.role,
            'summary': {
                'peak_memory_mb': self.peak_memory_mb,
                'bytes_sent': self.bytes_sent,
                'bytes_received': self.bytes_received,
                'comm_rounds': self.comm_rounds,
                'forward_passes': self.forward_passes,
                'backward_passes': self.backward_passes,
                'phase': {
                    'train': {
                        'peak_memory_mb': self.phase_peak_memory_mb.get('train', 0.0),
                        'bytes_sent': self.phase_bytes_sent.get('train', 0),
                        'bytes_received': self.phase_bytes_received.get('train', 0),
                        'comm_rounds': self.phase_comm_rounds.get('train', 0),
                    },
                    'eval': {
                        'peak_memory_mb': self.phase_peak_memory_mb.get('eval', 0.0),
                        'bytes_sent': self.phase_bytes_sent.get('eval', 0),
                        'bytes_received': self.phase_bytes_received.get('eval', 0),
                        'comm_rounds': self.phase_comm_rounds.get('eval', 0),
                    }
                }
            },
            'steps': self.step_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"[{self.role}] Stats saved to {filepath}")