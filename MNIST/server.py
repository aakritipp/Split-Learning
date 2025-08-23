import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from SGDGradientEst import StochasticGradientApproximator
import argparse


def get_dataloaders(train_batch_size, test_batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader


class ServerMNISTModel(nn.Module):
    def __init__(self):
        super(ServerMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        return x


class FullMNISTClassifier(nn.Module):
    def __init__(self):
        super(FullMNISTClassifier, self).__init__()
        # Server layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Client layers  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.out = nn.Linear(64 * 7 * 7, 10)
        
    def forward(self, x):
        # Server forward
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Client forward
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class Trainer:
    def __init__(self, conn):
        self.conn = conn
    
    def send_data(self, data):
        serialized = pickle.dumps(data)
        self.conn.sendall(len(serialized).to_bytes(4, 'big'))
        self.conn.sendall(serialized)
    
    def receive_data(self):
        length = int.from_bytes(self.conn.recv(4), 'big')
        data = b''
        while len(data) < length:
            data += self.conn.recv(length - len(data))
        return pickle.loads(data)


def calculate_metrics(outputs, labels, criterion):
    loss = criterion(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == labels).float().mean()
    return loss.item(), accuracy.item()


def train_distributed_model(server_model, full_model, train_loader, criterion, optimizer, 
                          scheduler, grad_estimator, use_zeroth_order_server, use_zeroth_order_client, trainer, device):
    """Distributed training function"""
    server_model.train()
    losses = []
    accs = []
    
    desc = f"Training (Server: {'ZOO' if use_zeroth_order_server else 'SGD'}, Client: {'ZOO' if use_zeroth_order_client else 'SGD'}):"
    
    with tqdm(total=len(train_loader), desc=desc) as t:
        for i, (images, labels) in enumerate(train_loader):
            try:
                if device != torch.device("cpu"):
                    images, labels = images.to(device), labels.to(device)
                
                if use_zeroth_order_server:
                    # Server uses ZOO - get client weights first, then do ZOO
                    with torch.no_grad():
                        optimizer.zero_grad()
                        
                        # Get current client weights before ZOO
                        trainer.send_data({'type': 'get_weights'})
                        client_weights = trainer.receive_data()['weights']
                        
                        def objective_fn(x, y):
                            # Use current server weights + client weights for full model
                            full_model.conv1.load_state_dict(server_model.conv1.state_dict())
                            full_model.conv2.load_state_dict(client_weights['conv2'])
                            full_model.out.load_state_dict(client_weights['out'])
                            
                            outputs = full_model(x)
                            loss = criterion(outputs, y)
                            return loss
                        
                        # Do ZOO gradient estimation
                        if grad_estimator is not None:
                            grad_estimator.estimate_gradients(
                                images, labels, objective_fn, random_seed=i**2 + i
                            )
                        optimizer.step()
                        
                        # Send forward pass result to client
                        server_output = server_model(images)
                        if use_zeroth_order_client:
                            # Client uses ZOO
                            trainer.send_data({
                                'type': 'forward_zoo', 
                                'data': server_output, 
                                'labels': labels,
                                'iteration': i
                            })
                            client_response = trainer.receive_data()
                        else:
                            # Client uses SGD
                            trainer.send_data({
                                'type': 'forward_sgd',
                                'data': server_output,
                                'labels': labels,
                                'iteration': i
                            })
                            client_response = trainer.receive_data()
                else:
                    # Server uses SGD - send batch first
                    trainer.send_data({
                        'images': images,
                        'labels': labels,
                        'iteration': i,
                        'use_zeroth_order_client': use_zeroth_order_client
                    })
                    
                    # Standard SGD for server
                    optimizer.zero_grad()
                    
                    # Forward through server
                    server_output = server_model(images)
                    server_output.requires_grad_(True)
                    
                    # Send to client for completion
                    trainer.send_data({'type': 'forward', 'data': server_output})
                    client_response = trainer.receive_data()
                    
                    # Get loss from client
                    loss = client_response['loss']
                    
                    if use_zeroth_order_client:
                        # Client uses ZOO - server computes gradients through full model
                        trainer.send_data({'type': 'get_weights'})
                        client_weights = trainer.receive_data()['weights']
                        
                        # Update full model and compute gradients
                        full_model.conv1.load_state_dict(server_model.conv1.state_dict())
                        full_model.conv2.load_state_dict(client_weights['conv2'])
                        full_model.out.load_state_dict(client_weights['out'])
                        
                        outputs = full_model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        
                        # Copy gradients to server model
                        for server_param, full_param in zip(server_model.parameters(), full_model.conv1.parameters()):
                            if full_param.grad is not None:
                                server_param.grad = full_param.grad.clone()
                        
                        full_model.zero_grad()
                    else:
                        # Client uses SGD - use gradients from client
                        server_grad = client_response['server_grad']
                        
                        # Apply gradients to server model
                        param_idx = 0
                        for param in server_model.parameters():
                            param_size = param.numel()
                            param.grad = server_grad[param_idx:param_idx + param_size].view(param.shape)
                            param_idx += param_size
                    
                    optimizer.step()
                
                # Calculate metrics using full model
                with torch.no_grad():
                    # Get current weights
                    trainer.send_data({'type': 'get_weights'})
                    client_weights = trainer.receive_data()['weights']
                    
                    # Update full model
                    full_model.conv1.load_state_dict(server_model.conv1.state_dict())
                    full_model.conv2.load_state_dict(client_weights['conv2'])
                    full_model.out.load_state_dict(client_weights['out'])
                    
                    outputs = full_model(images)
                    loss_val, accuracy = calculate_metrics(outputs, labels, criterion)
                    losses.append(loss_val)
                    accs.append(accuracy)
                
                # Update progress bar
                avg_loss = sum(losses) / len(losses)
                avg_acc = sum(accs) / len(accs)
                t.set_postfix({
                    "Loss": f"{avg_loss:.4f}",
                    "Accuracy": f"{avg_acc:.4f}"
                })
                t.update(1)
                
            except Exception as e:
                print(f"Error in server iteration {i}: {e}")
                losses.append(0.0)
                accs.append(0.0)
                t.update(1)
                break
        
        scheduler.step()
    
    # Return averages
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_acc = sum(accs) / len(accs) if accs else 0.0
    return avg_loss, avg_acc


def eval_model(full_model, server_model, test_loader, criterion, trainer, device):
    """Evaluate the distributed model"""
    full_model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            if device != torch.device("cpu"):
                images, labels = images.to(device), labels.to(device)
            
            # Get client weights
            trainer.send_data({'type': 'get_weights'})
            client_weights = trainer.receive_data()['weights']
            
            # Update full model
            full_model.conv1.load_state_dict(server_model.conv1.state_dict())
            full_model.conv2.load_state_dict(client_weights['conv2'])
            full_model.out.load_state_dict(client_weights['out'])
            
            outputs = full_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

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
    # Configuration
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch_dtype = torch.float32
    use_zeroth_order_server = args.use_zeroth_order

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("----- Using GPU -----")
    else:
        device = torch.device("cpu")
        print("----- Using CPU -----")
    
    # Model setup
    server_model = ServerMNISTModel().to(torch_dtype).to(device)
    full_model = FullMNISTClassifier().to(torch_dtype).to(device)
    
    # Initialize full model with server model weights
    full_model.conv1.load_state_dict(server_model.conv1.state_dict())
    
    print(f"Server total trainable parameters: {sum(p.numel() for p in server_model.parameters() if p.requires_grad)}")
    
    # Data setup
    train_loader, test_loader = get_dataloaders(args.train_batch_size, args.test_batch_size)
    
    # Optimizer setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(server_model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # ZOO setup
    grad_estimator = None
    if use_zeroth_order_server:
        def get_trainable_model_parameters(model):
            for param in model.parameters():
                if param.requires_grad:
                    yield param
        
        trainable_model_parameters = list(get_trainable_model_parameters(server_model))
        grad_estimator = StochasticGradientApproximator(
            model_params=trainable_model_parameters,
            perturbation_scale=args.mu,
            sample_count=args.num_pert,
            compute_device=device,
            data_type=torch_dtype
        )
    
    # Network setup
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('localhost', 12345))
    server_socket.listen(1)
    
    print("Server listening on localhost:12345")
    
    try:
        conn, addr = server_socket.accept()
        print(f"Client connected from {addr}")
        
        trainer = Trainer(conn)

        client_config = trainer.receive_data()
        use_zeroth_order_client = client_config['use_zeroth_order_client']

        print(f"Server using {'Zeroth-Order' if use_zeroth_order_server else 'Standard SGD'} optimization")
        print(f"Client using {'Zeroth-Order' if use_zeroth_order_client else 'Standard SGD'} optimization")
        
        # Training loop
        for epoch in range(args.epochs):
            train_loss, train_accuracy = train_distributed_model(
                server_model, full_model, train_loader, criterion, 
                optimizer, scheduler, grad_estimator,
                use_zeroth_order_server, use_zeroth_order_client, trainer, device
            )
            
            eval_loss, eval_accuracy = eval_model(
                full_model, server_model, test_loader, criterion, trainer, device
            )
            
            print(f"Evaluation(round {epoch}): Eval Loss:{eval_loss:.4f}, "
                  f"Accuracy:{eval_accuracy * 100:.2f}%")
        
        # Send completion signal
        trainer.send_data({'type': 'training_complete'})
        
    except Exception as e:
        print(f"Server error during training: {e}")
        try:
            trainer.send_data({'type': 'training_complete'})
        except:
            pass
    finally:
        try:
            conn.close()
        except:
            pass
        server_socket.close()
        print("Server shutdown complete")