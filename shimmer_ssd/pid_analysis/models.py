import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools
from typing import List, Callable
from contextlib import nullcontext

# Import AMP functionality
try:
    from torch.amp import autocast, GradScaler
    
    class DummyAMPModule:
        def __init__(self):
            self.autocast = autocast
            self.GradScaler = GradScaler
    
    amp = DummyAMPModule()
except ImportError:
    # Fallback for older PyTorch versions
    from contextlib import nullcontext
    
    class DummyAMPModule:
        def __init__(self):
            self.autocast = nullcontext
            self.GradScaler = None
    
    amp = DummyAMPModule()

from .sinkhorn import sinkhorn_probs

# Global configurations (avoid circular imports by defining locally)
global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False  # Default value
PRECISION = torch.float16  # Default precision for AMP
CHUNK_SIZE = 128  # Default chunk size
AGGRESSIVE_CLEANUP = False  # Whether to aggressively clean memory
MEMORY_CLEANUP_INTERVAL = 10  # Clean memory every N chunks

def mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
    activation: str
) -> nn.Sequential:
    act_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh
    }

    if activation not in act_map:
        raise ValueError(f"Unsupported activation: {activation}. Choose from {list(act_map.keys())}")
    
    act_layer = act_map[activation]
    modules = [
        nn.Linear(input_dim, hidden_dim),
        act_layer()
    ]
    for _ in range(layers):
        modules.extend([
            nn.Linear(hidden_dim, hidden_dim),
            act_layer()
        ])
    modules.append(nn.Linear(hidden_dim, output_dim))
    # Layers are moved to device when the model containing the mlp is moved.
    # Or, ensure .to(global_device) is called if mlp is used standalone.
    return nn.Sequential(*modules) #.to(global_device) -> model that uses it will be moved to device #TODO check if this is correct

class Discrim(nn.Module):
    def __init__(
        self,
        x_dim: int,
        hidden_dim: int,
        num_labels: int,
        layers: int,
        activation: str
    ):
        super().__init__()
        # mlp will be on CPU initially, then moved to device when Discrim instance is moved
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)
        
    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        # xs are expected to be on the same device as the model
        x = torch.cat(xs, dim=-1)
        return self.mlp(x)

class PretrainedDiscrim(nn.Module):
    """Discriminator using pretrained frozen encoder with additional softmax layers."""
    
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        num_labels: int,
        hidden_dim: int = 64,
        layers: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Store the pretrained encoder and freeze it
        self.pretrained_encoder = pretrained_encoder
        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False
        self.pretrained_encoder.eval()
        
        # Get the output dimension of the pretrained encoder
        # We'll infer this during the first forward pass
        self.encoder_output_dim = None
        self.classifier = None
        
        # Store parameters for lazy initialization
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.activation = activation
    
    def _initialize_classifier(self, encoder_output_dim: int):
        """Initialize the classifier MLP once we know the encoder output dimension."""
        if self.layers == 1:
            # Single layer: direct classification
            self.classifier = nn.Linear(encoder_output_dim, self.num_labels)
        else:
            # Two-layer MLP as requested
            act_map = {'relu': nn.ReLU, 'tanh': nn.Tanh}
            act_layer = act_map.get(self.activation, nn.ReLU)
            
            modules = [ #TODO this is not written generally, it is hardcoded for 2 layers. Use MLP for it, but be careful with dim matching.
                nn.Linear(encoder_output_dim, self.hidden_dim),
                act_layer(),
                nn.Linear(self.hidden_dim, self.num_labels)
            ]
            self.classifier = nn.Sequential(*modules)
        
        # Move to the same device as the encoder
        device = next(self.pretrained_encoder.parameters()).device
        self.classifier = self.classifier.to(device)
    
    def trainable_parameters(self):
        """Return only the parameters of the classifier (not the frozen encoder)."""
        if self.classifier is not None:
            return self.classifier.parameters()
        else:
            # Return empty generator if classifier not initialized yet
            return iter([])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through frozen pretrained encoder
        with torch.no_grad():
            encoded = self.pretrained_encoder(x)
        
        # Initialize classifier if not done yet
        if self.classifier is None:
            self.encoder_output_dim = encoded.size(-1)
            self._initialize_classifier(self.encoder_output_dim)
        
        # Pass through trainable classifier
        logits = self.classifier(encoded)
        return logits

class PretrainedJointDiscrim(nn.Module):
    """Joint discriminator that concatenates outputs from two pretrained encoders."""
    
    def __init__(
        self,
        pretrained_encoder1: nn.Module,
        pretrained_encoder2: nn.Module,
        num_labels: int,
        hidden_dim: int = 64,
        layers: int = 2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        # Store the pretrained encoders and freeze them
        self.pretrained_encoder1 = pretrained_encoder1
        self.pretrained_encoder2 = pretrained_encoder2
        
        for encoder in [self.pretrained_encoder1, self.pretrained_encoder2]:
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()
        
        # Get the combined output dimension of the pretrained encoders
        # We'll infer this during the first forward pass
        self.combined_output_dim = None
        self.classifier = None
        
        # Store parameters for lazy initialization
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.activation = activation
    
    def _initialize_classifier(self, combined_output_dim: int):
        """Initialize the classifier MLP once we know the combined encoder output dimension."""
        if self.layers == 1:
            # Single layer: direct classification
            self.classifier = nn.Linear(combined_output_dim, self.num_labels)
        else:
            # Two-layer MLP as requested #TODO this is not written generally, it is hardcoded for 2 layers. Use MLP for it, but be careful with dim matching.
            act_map = {'relu': nn.ReLU, 'tanh': nn.Tanh}
            act_layer = act_map.get(self.activation, nn.ReLU)
            
            modules = [
                nn.Linear(combined_output_dim, self.hidden_dim),
                act_layer(),
                nn.Linear(self.hidden_dim, self.num_labels)
            ]
            self.classifier = nn.Sequential(*modules)
        
        # Move to the same device as the encoders
        device = next(self.pretrained_encoder1.parameters()).device
        self.classifier = self.classifier.to(device)
    
    def trainable_parameters(self):
        """Return only the parameters of the classifier (not the frozen encoders)."""
        if self.classifier is not None:
            return self.classifier.parameters()
        else:
            # Return empty generator if classifier not initialized yet
            return iter([])
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Pass through frozen pretrained encoders
        with torch.no_grad():
            encoded1 = self.pretrained_encoder1(x1)
            encoded2 = self.pretrained_encoder2(x2)
        
        # Concatenate the encoded representations
        combined = torch.cat([encoded1, encoded2], dim=-1)
        
        # Initialize classifier if not done yet
        if self.classifier is None:
            self.combined_output_dim = combined.size(-1)
            self._initialize_classifier(self.combined_output_dim)
        
        # Pass through trainable classifier
        logits = self.classifier(combined)
        return logits

class CEAlignment(nn.Module): #TODO give another name to this class
    def __init__(
        self,
        x1_dim: int,
        x2_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_labels: int,
        layers: int,
        activation: str
    ):
        super().__init__()
        self.num_labels = num_labels
        # mlp1 and mlp2 will be on CPU initially, then moved to device with CEAlignment instance
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation) #TODO think carefully about the scaling as the num clusters increases... do we have enough data?
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        #TODO find out if the embed dim times num labels is correct. On a general level, code is fine, but approach may not be.
        #TODO find out where the embed dim comes from. And generally the params of these networks, I think it is not CLI, so it must be some arbitrary default in main.py
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        p_y_x1: torch.Tensor,
        p_y_x2: torch.Tensor,
        lr_finding_mode: bool = False
    ) -> torch.Tensor:
        batch_size = x1.size(0)
        if x2.size(0) != batch_size:
            raise ValueError(f"Batch size mismatch: x1 has batch size {batch_size} but x2 has batch size {x2.size(0)}. Inputs must have matching batch dimensions.")
        
        q1 = self.mlp1(x1).unflatten(1, (self.num_labels, -1)) 
        q2 = self.mlp2(x2).unflatten(1, (self.num_labels, -1))

        q1 = (q1 - q1.mean(dim=2, keepdim=True)) / torch.sqrt(q1.var(dim=2, keepdim=True) + 1e-8)
        q2 = (q2 - q2.mean(dim=2, keepdim=True)) / torch.sqrt(q2.var(dim=2, keepdim=True) + 1e-8)
        #TODO study how einsum works in depth
        aff = torch.einsum('bce, dce -> bdc', q1, q2) / math.sqrt(q1.size(-1))
        aff_max = aff.reshape(-1, aff.size(-1)).max(dim=0)[0]
        aff_centered = aff - aff_max.view(1, 1, -1) #TODO check if they also do this in the paper, if not investigate more
        A = torch.exp(aff_centered)

        # Create a combined model that includes both mlp1 and mlp2 for gradient tracking (always available)
        combined_params = {}
        combined_params.update({f"mlp1.{name}": param for name, param in self.mlp1.named_parameters()})
        combined_params.update({f"mlp2.{name}": param for name, param in self.mlp2.named_parameters()})

        # Capture gradient magnitudes before Sinkhorn operations (only when not in LR finding mode)
        if not lr_finding_mode:
            try:
                from .coupling_visualization import capture_gradient_magnitudes
                # Capture gradients before Sinkhorn
                gradients_before = {}
                for param_name, param in combined_params.items():
                    if param.grad is not None:
                        gradients_before[param_name] = param.grad.norm().item()
            except ImportError:
                gradients_before = {}
        else:
            gradients_before = {}

        couplings = []
        for c in range(self.num_labels):
            # PAPER'S APPROACH: Use separate marginals directly as per Algorithm 1
            # Row marginal = p(y=c | x1), Column marginal = p(y=c | x2)
            coupling_c = sinkhorn_probs(
                A[..., c],
                p_y_x1[:, c],  # row-marginal = p(y=c | x1) - no averaging
                p_y_x2[:, c],  # col-marginal = p(y=c | x2) - no averaging
                # Potentially pass chunk_size from this model if needed by sinkhorn
                # chunk_size = self.chunk_size # If CEAlignment had self.chunk_size
                log_to_wandb=True,  # Enable wandb logging for coupling matrices
                wandb_prefix=f"coupling_cluster_{c}",  # Log each cluster separately
                wandb_log_interval=25,  # Log every 25 iterations for more granular tracking
                lr_finding_mode=lr_finding_mode,  # Skip expensive visualizations during LR finding
            )
            couplings.append(coupling_c)
        
        # Capture gradient magnitudes after Sinkhorn operations and log comparison
        if not lr_finding_mode and gradients_before:
            try:
                from .coupling_visualization import log_gradient_magnitudes
                
                # Capture gradients after Sinkhorn  
                gradients_after = {}
                for param_name, param in combined_params.items():
                    if param.grad is not None:
                        gradients_after[param_name] = param.grad.norm().item()
                
                # Log gradient magnitude changes to wandb
                if gradients_after:  # Only log if we have gradients
                    log_gradient_magnitudes(
                        parameters_before=gradients_before,
                        parameters_after=gradients_after,
                        step=None,  # Could be passed as parameter if needed
                        prefix="gradient_magnitudes",
                        lr_finding_mode=lr_finding_mode
                    )
            except ImportError:
                pass  # Gracefully handle missing visualization module
        
        P = torch.stack(couplings, dim=-1)
        return P, A  # Return both coupling matrix and affinity matrix for debugging

class CEAlignmentInformation(nn.Module): #TODO give another name to this class
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, discrim_1, discrim_2, discrim_12, p_y):
        
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        
        # Store discriminators
        self.discrim_1 = discrim_1
        self.discrim_2 = discrim_2
        self.discrim_12 = discrim_12
        
        # Freeze pre-trained discriminators
        for D in (self.discrim_1, self.discrim_2, self.discrim_12):
            D.eval()
            for p in D.parameters():
                p.requires_grad = False
                
        # marginal p(y)
        self.register_buffer('p_y', p_y) #TODO where do we calculate Py? need to plot it, also.
        #TODO find out if mixed precision can be savely removed
        # For mixed precision - import USE_AMP from utils
        try:
            from .utils import USE_AMP, CHUNK_SIZE, AGGRESSIVE_CLEANUP, MEMORY_CLEANUP_INTERVAL
            self.use_amp = USE_AMP
            self.chunk_size = CHUNK_SIZE
            self.aggressive_cleanup = AGGRESSIVE_CLEANUP
            self.memory_cleanup_interval = MEMORY_CLEANUP_INTERVAL
        except ImportError:
            # Fallback to default values
            self.use_amp = False
            self.chunk_size = 128
            self.aggressive_cleanup = False
            self.memory_cleanup_interval = 10
        
        # Create GradScaler for mixed precision training
        if self.use_amp:
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
            except ImportError:
                self.use_amp = False
                self.scaler = None
        else:
            self.scaler = None
            
        # Flag to control visualization during LR finding
        self.lr_finding_mode = False
        
        # 🔬 DEBUG FLAGS - Control debugging features
        self.debug_gradients = False
        self.debug_weights = False  
        self.debug_numerical = False
        self.debug_verbose = False
        self.debug_interval = 100
        self.debug_step_counter = 0

    def set_lr_finding_mode(self, enabled: bool):
        """Enable or disable LR finding mode to control visualization."""
        self.lr_finding_mode = enabled
    
    def configure_debugging(self, debug_gradients: bool = False, debug_weights: bool = False, 
                           debug_numerical: bool = False, debug_verbose: bool = False,
                           debug_interval: int = 100):
        """
        Configure debugging settings for the model.
        
        Args:
            debug_gradients: Enable gradient flow inspection
            debug_weights: Enable weight update checking  
            debug_numerical: Enable numerical range inspection
            debug_verbose: Enable verbose debug output
            debug_interval: Interval for debug checks (every N steps)
        """
        self.debug_gradients = debug_gradients
        self.debug_weights = debug_weights
        self.debug_numerical = debug_numerical
        self.debug_verbose = debug_verbose
        self.debug_interval = debug_interval
        self.debug_step_counter = 0
        
        if debug_gradients or debug_weights or debug_numerical:
            print("🔬 Debugging enabled:")
            if debug_gradients:
                print("  ✓ Gradient flow inspection")
            if debug_weights:
                print("  ✓ Weight update checking")
            if debug_numerical:
                print("  ✓ Numerical range inspection")
            print(f"  📊 Debug interval: every {debug_interval} steps")
            if debug_verbose:
                print("  📝 Verbose output enabled")
    
    def inspect_alignment_gradients(self, print_details: bool = True):
        """
        Inspect gradients flowing into alignment MLPs after loss.backward().
        Call this method after loss.backward() to check if Sinkhorn blocks autograd.
        
        Args:
            print_details: Whether to print gradient norms to console
            
        Returns:
            Dict with gradient information for debugging
        """
        grad_info = {
            'mlp1_gradients': {},
            'mlp2_gradients': {},
            'gradients_present': False,
            'max_grad_norm': 0.0
        }
        
        # Check MLP1 gradients
        if print_details:
            print("=== MLP1 Gradient Inspection ===")
        for name, param in self.align.mlp1.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_info['mlp1_gradients'][name] = grad_norm
                grad_info['gradients_present'] = True
                grad_info['max_grad_norm'] = max(grad_info['max_grad_norm'], grad_norm)
                if print_details:
                    print(f"  {name}: grad_norm = {grad_norm:.6f}")
            else:
                grad_info['mlp1_gradients'][name] = None
                if print_details:
                    print(f"  {name}: grad = None")
        
        # Check MLP2 gradients  
        if print_details:
            print("=== MLP2 Gradient Inspection ===")
        for name, param in self.align.mlp2.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_info['mlp2_gradients'][name] = grad_norm
                grad_info['gradients_present'] = True
                grad_info['max_grad_norm'] = max(grad_info['max_grad_norm'], grad_norm)
                if print_details:
                    print(f"  {name}: grad_norm = {grad_norm:.6f}")
            else:
                grad_info['mlp2_gradients'][name] = None
                if print_details:
                    print(f"  {name}: grad = None")
        
        # Summary
        if print_details:
            print("=== Gradient Flow Summary ===")
            if grad_info['gradients_present']:
                print(f"✓ Gradients are flowing! Max norm: {grad_info['max_grad_norm']:.6f}")
                if grad_info['max_grad_norm'] < 1e-6:
                    print("⚠️  Warning: Gradient norms are very small - check for vanishing gradients")
                elif grad_info['max_grad_norm'] > 10.0:
                    print("⚠️  Warning: Gradient norms are large - check for exploding gradients")
            else:
                print("✗ NO GRADIENTS FOUND - Sinkhorn may be blocking autograd!")
                print("   This indicates the Sinkhorn implementation is not differentiable")
        
        return grad_info
    
    def inspect_numerical_ranges(self, A, P, log_ratio, print_details: bool = True):
        """
        Inspect numerical ranges of affinity matrix, coupling matrix, and log_ratio.
        Call this to detect overflow/underflow issues that could cause NaN or infinite values.
        
        Args:
            A: Affinity matrix from CEAlignment
            P: Coupling matrix from Sinkhorn
            log_ratio: Log ratio for MI calculation
            print_details: Whether to print ranges to console
            
        Returns:
            Dict with numerical range information for debugging
        """
        range_info = {
            'A_range': (A.min().item(), A.max().item()),
            'P_range': (P.min().item(), P.max().item()),
            'log_ratio_range': (log_ratio.min().item(), log_ratio.max().item()),
            'has_inf': False,
            'has_nan': False,
            'numerical_issues': []
        }
        
        # Check for infinities and NaNs
        if torch.isinf(A).any():
            range_info['has_inf'] = True
            range_info['numerical_issues'].append('A contains infinities')
        if torch.isnan(A).any():
            range_info['has_nan'] = True
            range_info['numerical_issues'].append('A contains NaNs')
            
        if torch.isinf(P).any():
            range_info['has_inf'] = True
            range_info['numerical_issues'].append('P contains infinities')
        if torch.isnan(P).any():
            range_info['has_nan'] = True
            range_info['numerical_issues'].append('P contains NaNs')
            
        if torch.isinf(log_ratio).any():
            range_info['has_inf'] = True
            range_info['numerical_issues'].append('log_ratio contains infinities')
        if torch.isnan(log_ratio).any():
            range_info['has_nan'] = True
            range_info['numerical_issues'].append('log_ratio contains NaNs')
        
        if print_details:
            print("=== Numerical Range Inspection ===")
            print(f"A range: [{range_info['A_range'][0]:.6f}, {range_info['A_range'][1]:.6f}]")
            print(f"P range: [{range_info['P_range'][0]:.6f}, {range_info['P_range'][1]:.6f}]")
            print(f"log_ratio range: [{range_info['log_ratio_range'][0]:.6f}, {range_info['log_ratio_range'][1]:.6f}]")
            
            # Check for extreme values that could cause issues
            A_min, A_max = range_info['A_range']
            if A_max > 50:
                print(f"⚠️  Warning: A_max = {A_max:.2f} is very large (risk of exp overflow)")
            if A_min < -50:
                print(f"⚠️  Warning: A_min = {A_min:.2f} is very negative (risk of exp underflow)")
                
            P_min, P_max = range_info['P_range']
            if P_min < 1e-10:
                print(f"⚠️  Warning: P_min = {P_min:.2e} is very small (risk of log underflow)")
            if P_max > 1e10:
                print(f"⚠️  Warning: P_max = {P_max:.2e} is very large")
                
            lr_min, lr_max = range_info['log_ratio_range']
            if abs(lr_min) > 100 or abs(lr_max) > 100:
                print(f"⚠️  Warning: log_ratio has extreme values (risk of numerical instability)")
            
            # Report issues
            if range_info['numerical_issues']:
                print("✗ NUMERICAL ISSUES DETECTED:")
                for issue in range_info['numerical_issues']:
                    print(f"  - {issue}")
            else:
                print("✓ No infinities or NaNs detected")
        
        return range_info
    
    def capture_alignment_weights(self):
        """
        Capture current alignment MLP weights for before/after optimizer comparison.
        Call this before forward/backward/step to store weights for comparison.
        
        Returns:
            Dict containing cloned weight tensors for comparison
        """
        weight_snapshot = {
            'mlp1_weights': {},
            'mlp2_weights': {}
        }
        
        # Capture MLP1 weights
        for name, param in self.align.mlp1.named_parameters():
            weight_snapshot['mlp1_weights'][name] = param.clone().detach()
        
        # Capture MLP2 weights  
        for name, param in self.align.mlp2.named_parameters():
            weight_snapshot['mlp2_weights'][name] = param.clone().detach()
            
        return weight_snapshot
    
    def check_weight_updates(self, weights_before, print_details: bool = True):
        """
        Check if alignment MLP weights have been updated after optimizer.step().
        Call this after optimizer.step() with weights captured before training step.
        
        Args:
            weights_before: Dict from capture_alignment_weights() before optimizer.step()
            print_details: Whether to print update information to console
            
        Returns:
            Dict with weight update information for debugging
        """
        update_info = {
            'mlp1_updates': {},
            'mlp2_updates': {},
            'any_updates': False,
            'max_change': 0.0,
            'total_params_updated': 0,
            'total_params': 0
        }
        
        # Check MLP1 weight updates
        if print_details:
            print("=== MLP1 Weight Update Check ===")
        for name, param in self.align.mlp1.named_parameters():
            if name in weights_before['mlp1_weights']:
                before = weights_before['mlp1_weights'][name]
                after = param.detach()
                
                # Check if weights changed
                weights_changed = not torch.allclose(before, after, atol=1e-8)
                
                if weights_changed:
                    max_change = (after - before).abs().max().item()
                    mean_change = (after - before).abs().mean().item()
                    update_info['mlp1_updates'][name] = {
                        'changed': True,
                        'max_change': max_change,
                        'mean_change': mean_change
                    }
                    update_info['any_updates'] = True
                    update_info['max_change'] = max(update_info['max_change'], max_change)
                    update_info['total_params_updated'] += 1
                    
                    if print_details:
                        print(f"  {name}: ✓ UPDATED (max_change={max_change:.2e}, mean_change={mean_change:.2e})")
                else:
                    update_info['mlp1_updates'][name] = {'changed': False}
                    if print_details:
                        print(f"  {name}: ✗ NO CHANGE")
                
                update_info['total_params'] += 1
        
        # Check MLP2 weight updates
        if print_details:
            print("=== MLP2 Weight Update Check ===")
        for name, param in self.align.mlp2.named_parameters():
            if name in weights_before['mlp2_weights']:
                before = weights_before['mlp2_weights'][name]
                after = param.detach()
                
                # Check if weights changed
                weights_changed = not torch.allclose(before, after, atol=1e-8)
                
                if weights_changed:
                    max_change = (after - before).abs().max().item()
                    mean_change = (after - before).abs().mean().item()
                    update_info['mlp2_updates'][name] = {
                        'changed': True,
                        'max_change': max_change,
                        'mean_change': mean_change
                    }
                    update_info['any_updates'] = True
                    update_info['max_change'] = max(update_info['max_change'], max_change)
                    update_info['total_params_updated'] += 1
                    
                    if print_details:
                        print(f"  {name}: ✓ UPDATED (max_change={max_change:.2e}, mean_change={mean_change:.2e})")
                else:
                    update_info['mlp2_updates'][name] = {'changed': False}
                    if print_details:
                        print(f"  {name}: ✗ NO CHANGE")
                
                update_info['total_params'] += 1
        
        # Summary
        if print_details:
            print("=== Weight Update Summary ===")
            if update_info['any_updates']:
                print(f"✓ Weights are being updated! {update_info['total_params_updated']}/{update_info['total_params']} parameter groups changed")
                print(f"  Max change magnitude: {update_info['max_change']:.2e}")
                if update_info['max_change'] < 1e-8:
                    print("  ⚠️  Warning: Changes are very small - check learning rate")
                elif update_info['max_change'] > 1.0:
                    print("  ⚠️  Warning: Changes are very large - check learning rate/gradients")
            else:
                print("✗ NO WEIGHT UPDATES DETECTED!")
                print("  Possible causes:")
                print("  - Alignment parameters not added to optimizer")
                print("  - optimizer.zero_grad() not called")
                print("  - optimizer.step() not called")
                print("  - No gradients flowing (check with inspect_alignment_gradients)")
                print("  - Learning rate is zero")
        
        return update_info
        
    def quick_weight_update_check(self, print_details: bool = True):
        """
        Simplified check for a single parameter to quickly verify optimizer is working.
        This mimics the user's suggested check: w0 = param.clone(); ...; torch.allclose(w0, param)
        
        Returns:
            Boolean indicating if weights changed (True = good, False = problem)
        """
        # Get first parameter from MLP1
        first_param = next(self.align.mlp1.parameters())
        w0 = first_param.clone().detach()
        
        if print_details:
            print("=== Quick Weight Update Check ===")
            print("Captured first MLP1 parameter before optimization step")
            print("Run your forward/backward/step cycle, then call this method again with check_after=True")
        
        return w0
    
    def quick_weight_update_verify(self, w0, print_details: bool = True):
        """
        Verify if weights changed after optimization step.
        
        Args:
            w0: Weight tensor captured before optimization (from quick_weight_update_check)
            
        Returns:
            Boolean: True if weights changed (good), False if no change (problem)
        """
        first_param = next(self.align.mlp1.parameters())
        weights_changed = not torch.allclose(w0, first_param, atol=1e-8)
        
        if print_details:
            print("=== Quick Weight Update Verification ===")
            if weights_changed:
                max_change = (first_param - w0).abs().max().item()
                print(f"✓ Weights CHANGED! Max change: {max_change:.2e}")
                print("  Optimizer is working correctly")
            else:
                print("✗ Weights DID NOT CHANGE!")
                print("  Problem: Optimizer is not updating alignment weights")
                print("  Check: alignment parameters in optimizer, zero_grad(), step() calls")
        
        return weights_changed
    
    def debug_training_step(self, before_backward: bool = True, weights_before=None, optimizer=None):
        """
        Perform debugging checks during training if enabled.
        Call this method at different points in your training loop.
        
        Args:
            before_backward: If True, captures weights before backward pass
                           If False, checks gradients and weight updates after step
            weights_before: Weights captured before training step (for after checks)
            optimizer: Optimizer instance (for verification)
            
        Returns:
            weights_before dict if before_backward=True, update info dict if before_backward=False
        """
        # Skip if debugging disabled or not at debug interval
        if not (self.debug_gradients or self.debug_weights):
            return None
            
        if self.debug_step_counter % self.debug_interval != 0:
            return None
        
        if before_backward:
            # Capture weights before training step
            if self.debug_weights:
                return self.capture_alignment_weights()
            return None
        else:
            # Check gradients and weight updates after backward/step
            results = {}
            
            if self.debug_gradients:
                print(f"\n🔬 [Step {self.debug_step_counter}] Gradient Flow Check:")
                grad_info = self.inspect_alignment_gradients(print_details=self.debug_verbose)
                results['gradients'] = grad_info
            
            if self.debug_weights and weights_before is not None:
                print(f"\n🔬 [Step {self.debug_step_counter}] Weight Update Check:")
                update_info = self.check_weight_updates(weights_before, print_details=self.debug_verbose)
                results['weights'] = update_info
                
                # Quick summary for non-verbose mode
                if not self.debug_verbose and update_info['any_updates']:
                    print(f"✅ {update_info['total_params_updated']}/{update_info['total_params']} parameters updated (max change: {update_info['max_change']:.2e})")
                elif not self.debug_verbose and not update_info['any_updates']:
                    print("❌ No weight updates detected!")
            
            return results
        
    def debug_quick_check(self):
        """
        Quick debugging check for immediate use - mimics user's suggested approach.
        Returns weight tensor to check after optimizer step.
        """
        if not self.debug_weights:
            return None
        return self.quick_weight_update_check(print_details=self.debug_verbose)
    
    def debug_quick_verify(self, w0):
        """
        Quick verification of weight changes - mimics user's suggested approach.
        """
        if not self.debug_weights or w0 is None:
            return True  # Assume success if debugging disabled
        return self.quick_weight_update_verify(w0, print_details=self.debug_verbose)
        
    def forward(self, x1, x2, y):
        """
        Compute PID components between domain representations.
        
        Args:
            x1: First domain features, shape [batch_size, x1_dim]
            x2: Second domain features, shape [batch_size, x2_dim]
            y: Labels, shape [batch_size]
            
        Returns:
            Tuple of (loss, pid_vals, P) where:
            - loss: Scalar tensor for optimization
            - pid_vals: Tensor of shape [4] containing [redundancy, unique1, unique2, synergy]
            - P: Coupling matrix
        """
        batch_size = x1.size(0)
        
        # Forward pass with mixed precision if enabled
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Ensure nullcontext is available in this scope
        try:
            from torch.amp import autocast
        except ImportError:
            # Fallback for older PyTorch versions
            autocast = nullcontext  # Use the nullcontext imported at module level
        
        with autocast(device_type) if self.use_amp else nullcontext():
            # Get label conditional probabilities using discriminators with proper softmax
            p_y_x1 = F.softmax(self.discrim_1(x1), dim=1)  # [batch, num_labels]
            p_y_x2 = F.softmax(self.discrim_2(x2), dim=1)  # [batch, num_labels]
            
            # Handle joint discriminator based on its type
            if isinstance(self.discrim_12, PretrainedJointDiscrim):
                # PretrainedJointDiscrim expects two separate arguments
                p_y_x1x2 = F.softmax(self.discrim_12(x1, x2), dim=1)  # [batch, num_labels]
            else:
                # Regular Discrim expects concatenated input
                p_y_x1x2 = F.softmax(self.discrim_12(torch.cat([x1, x2], dim=-1)), dim=1)  # [batch, num_labels]
        
        # Calculate unimodal mutual information terms with explicit dtype for numerical stability
        mi_x1_y = torch.sum(p_y_x1 * torch.log(p_y_x1 / self.p_y.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        mi_x2_y = torch.sum(p_y_x2 * torch.log(p_y_x2 / self.p_y.unsqueeze(0) + 1e-8), dim=1, dtype=torch.float32)
        
        # Log discriminator marginal distributions to wandb
        try:
            from .coupling_visualization import log_discriminator_marginals_to_wandb
            log_discriminator_marginals_to_wandb(
                p_y_x1=p_y_x1,
                p_y_x2=p_y_x2,
                p_y_x1x2=p_y_x1x2,
                p_y_marginal=self.p_y,
                step=None,  # Could be passed as parameter if needed
                prefix="discriminator_marginals",
                cluster_names=None,  # Could be passed as parameter if available
                lr_finding_mode=self.lr_finding_mode
            )
        except ImportError:
            pass  # Gracefully handle missing visualization module
        
        # Get coupling matrix and affinity matrix from align (pass lr_finding_mode to control visualization)
        P, A = self.align(x1, x2, p_y_x1, p_y_x2, lr_finding_mode=self.lr_finding_mode)  # [batch, batch, num_labels]
        
        # 1) Normalize along the label axis to get q̃(y|x1,x2)
        P_cond = P / (P.sum(dim=-1, keepdim=True) + 1e-8)  # [batch, batch, num_labels]
        
        # 2) Compute the joint mutual information properly using p(y) as denominator
        # Expand p_y for broadcasting
        p_y_expanded = self.p_y.view(1, 1, -1)  # [1, 1, num_labels]
        
        # Calculate proper log ratio for joint MI
        log_ratio = torch.log(P_cond + 1e-8) - torch.log(p_y_expanded + 1e-8)  # [batch, batch, num_labels]
        
        # 🔬 DEBUGGING: Conditional numerical range inspection
        if self.debug_numerical and not self.lr_finding_mode and self.debug_step_counter % self.debug_interval == 0:
            try:
                self.inspect_numerical_ranges(A, P, log_ratio, print_details=self.debug_verbose)
            except Exception as e:
                print(f"⚠️  Numerical inspection failed: {e}")
        
        # Increment debug step counter
        self.debug_step_counter += 1
        
        # 3) Compute joint MI by summing over all dimensions, weighted by joint coupling P
        mi_x1x2_y = (P * log_ratio).sum(dim=[1, 2])  # [batch]
        
        # For comparison - calculate joint MI from discriminator (not used in updated PID calculation)
        mi_discrim_x1x2_y = torch.sum(p_y_x1x2 * torch.log(p_y_x1x2 / self.p_y.unsqueeze(0) + 1e-8), 
                                     dim=1, dtype=torch.float32)
        
        # 🔬 CRITICAL MATHEMATICAL CORRECTION: Proper PID computation on scalar MI terms
        # 1) Compute the global mutual information terms (scalars):
        I_x1 = mi_x1_y.mean()  # I(X1;Y)
        I_x2 = mi_x2_y.mean()  # I(X2;Y)
        I_q = mi_x1x2_y.mean()  # I_q(X1,X2;Y) from coupling
        I_p = mi_discrim_x1x2_y.mean()  # I_p(X1,X2;Y) from joint discriminator
        
        # 2) Apply the Möbius relations on scalars:
        # Redundancy = I(X1;Y) + I(X2;Y) - I_q(X1,X2;Y)
        redundancy = torch.clamp(I_x1 + I_x2 - I_q, min=0)
        # Unique1 = I(X1;Y) - Redundancy
        unique1 = torch.clamp(I_x1 - redundancy, min=0)
        # Unique2 = I(X2;Y) - Redundancy
        unique2 = torch.clamp(I_x2 - redundancy, min=0)
        # Synergy = I_p(X1,X2;Y) - I_q(X1,X2;Y)
        synergy = torch.clamp(I_p - I_q, min=0)
        
        loss = I_q  # Optimize the coupling-based joint MI
        
        # Final cleanup
        if self.aggressive_cleanup and torch.cuda.is_available():
            if torch.cuda.current_stream().query() and self.memory_cleanup_interval > 0:
                torch.cuda.empty_cache()
        
        # Return loss, PID components, and coupling matrix
        pid_vals = torch.stack([redundancy, unique1, unique2, synergy])
        return loss, pid_vals, P 