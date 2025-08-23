import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from SGDGradientEst import StochasticGradientApproximator
import argparse


class ClientMNISTModel(nn.Module):
    def __init__(self):
        super(ClientMNISTModel, self).__init__()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.out = nn.Linear(64 * 7 * 7, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class Client:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((host, port))
        
    def send_data(self, data):
        serialized = pickle.dumps(data)
        self.socket.sendall(len(serialized).to_bytes(4, 'big'))
        self.socket.sendall(serialized)
    
    def receive_data(self):
        length = int.from_bytes(self.socket.recv(4), 'big')
        data = b''
        while len(data) < length:
            data += self.socket.recv(length - len(data))
        return pickle.loads(data)
    
    def close(self):
        self.socket.close()


def handle_training_requests(client_model, criterion, optimizer, scheduler, 
                           grad_estimator, client, device):
    """Handle training requests from server"""
    
    # Initialize global variable for labels
    global current_labels
    current_labels = None
    
    while True:
        try:
            # Receive data from server
            server_data = client.receive_data()
            
            if server_data.get('type') == 'training_complete':
                print("Training completed. Client shutting down.")
                break
                
            elif server_data.get('type') == 'get_weights':
                # Send current model weights
                weights = {
                    'conv2': client_model.conv2.state_dict(),
                    'out': client_model.out.state_dict()
                }
                client.send_data({'weights': weights})
                
            elif server_data.get('type') == 'forward_zoo':
                # Handle forward pass when server uses ZOO and client uses ZOO
                server_output = server_data['data'].to(device)
                labels = server_data['labels'].to(device)
                iteration = server_data['iteration']
                
                # Client does its ZOO optimization
                client_model.train()
                with torch.no_grad():
                    optimizer.zero_grad()
                    
                    def client_objective_fn(dummy_input, target_labels):
                        client_output = client_model(server_output)
                        loss = criterion(client_output, target_labels)
                        return loss
                    
                    if grad_estimator is not None:
                        grad_estimator.estimate_gradients(
                            server_output, labels, client_objective_fn, random_seed=iteration**2 + iteration
                        )
                    optimizer.step()
                
                # Send back confirmation
                client.send_data({'status': 'updated'})
                
            elif server_data.get('type') == 'forward_sgd':
                # Handle forward pass when server uses ZOO and client uses SGD
                server_output = server_data['data'].to(device)
                labels = server_data['labels'].to(device)
                
                client_model.train()
                optimizer.zero_grad()
                
                server_output.requires_grad_(True)
                
                # Forward through client model
                client_output = client_model(server_output)
                loss = criterion(client_output, labels)
                
                # Backward pass
                loss.backward()
                
                # Update client parameters
                optimizer.step()
                
                # Send confirmation back to server
                client.send_data({'status': 'updated'})
                    
            elif server_data.get('type') == 'forward':
                # Handle forward pass when server uses SGD (labels should be available)
                server_output = server_data['data'].to(device)
                server_output.requires_grad_(True)
                
                # Labels should have been set from the batch message
                if current_labels is not None:
                    labels = current_labels
                else:
                    print("Error: No labels available for forward pass")
                    client.send_data({'error': 'no_labels'})
                    continue
                
                # Forward through client model
                client_output = client_model(server_output)
                loss = criterion(client_output, labels)
                
                # Backward pass
                loss.backward()
                
                # Send gradients back to server
                server_grad = server_output.grad.flatten() if server_output.grad is not None else torch.zeros(server_output.numel()).to(device)
                client.send_data({
                    'loss': loss.detach(),
                    'server_grad': server_grad
                })
                
                # Update client parameters
                optimizer.step()
                
            else:
                # Handle training batch (when server uses SGD)
                images = server_data['images'].to(device)
                labels = server_data['labels'].to(device)
                iteration = server_data['iteration']
                use_zeroth_order_client_from_server = server_data['use_zeroth_order_client']
                
                # Store labels for potential forward pass
                current_labels = labels
                
                if use_zeroth_order_client_from_server:
                    # Client uses ZOO - wait for server's forward pass
                    server_request = client.receive_data()
                    
                    if server_request['type'] == 'forward':
                        server_output = server_request['data'].to(device)
                        
                        # Do ZOO optimization
                        client_model.train()
                        with torch.no_grad():
                            optimizer.zero_grad()
                            
                            def client_objective_fn(dummy_input, target_labels):
                                client_output = client_model(server_output)
                                loss = criterion(client_output, target_labels)
                                return loss
                            
                            if grad_estimator is not None:
                                grad_estimator.estimate_gradients(
                                    images, labels, client_objective_fn, random_seed=iteration**2 + iteration
                                )
                            optimizer.step()
                        
                        # Send back the loss
                        with torch.no_grad():
                            client_output = client_model(server_output)
                            loss = criterion(client_output, labels)
                            client.send_data({'loss': loss})
                else:
                    # Client uses SGD - just wait for forward pass
                    # The forward pass will be handled by the 'forward' case above
                    pass
                        
        except Exception as e:
            print(f"Error in client training: {e}")
            break

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Split Learning MNIST Training')
    
    # Training hyperparameters
    parser.add_argument('--seed', type=int, default=365, help='Random seed for reproducibility')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    
    # Zeroth-order optimization parameters
    parser.add_argument('--mu', type=float, default=1e-3, help='Perturbation scale for zeroth-order optimization')
    parser.add_argument('--num_pert', type=int, default=10, help='Number of perturbations for gradient estimation')
    parser.add_argument('--use_zeroth_order', action='store_true', default=False, help='Use zeroth-order optimization')
    
    # Data and batch parameters
    parser.add_argument('--train_batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Global variable to store current labels
    current_labels = None
    args = parse_args()
    
    # Configuration (should match server)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch_dtype = torch.float32
    
    print(f"Client using {'Zeroth-Order' if args.use_zeroth_order else 'Standard SGD'} optimization")
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("----- Using GPU -----")
    else:
        device = torch.device("cpu")
        print("----- Using CPU -----")
    
    # Model setup
    client_model = ClientMNISTModel().to(torch_dtype).to(device)
    
    # Optimizer setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(client_model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # ZOO setup
    grad_estimator = None
    if args.use_zeroth_order:
        def get_trainable_model_parameters(model):
            for param in model.parameters():
                if param.requires_grad:
                    yield param
        
        trainable_model_parameters = list(get_trainable_model_parameters(client_model))
        grad_estimator = StochasticGradientApproximator(
            model_params=trainable_model_parameters,
            perturbation_scale=args.mu,
            sample_count=args.num_pert,
            compute_device=device,
            data_type=torch_dtype
        )
    
    try:
        # Connect to server
        client = Client('localhost', 12345)
        print("Connected to server at localhost:12345")

        client_config = {'use_zeroth_order_client': args.use_zeroth_order}
        client.send_data(client_config)
        print(f"Sent client configuration: use_zeroth_order_client={args.use_zeroth_order}")

        
        # Handle training requests
        handle_training_requests(
            client_model, criterion, optimizer, scheduler,
            grad_estimator,  # This will be None if not using zeroth-order
            client, device
        )
        
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        try:
            client.close()
        except:
            pass
        print("Client shutdown complete")