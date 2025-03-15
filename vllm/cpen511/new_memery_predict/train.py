
import torch
import torch.optim as optim
from dataloader import SequenceDataset
from model import MemoryPredict
import sys

windows_size = 50
file_path = "/root/vllm/vllm/cpen511/data/pure_sequence.csv"
output_log = "/root/vllm/vllm/cpen511/data/output.txt"

# Redirect output
result_output = open(output_log, "w", buffering=1)
sys.stdout = result_output


# Split into train and test
data = SequenceDataset(file_path , windows_size)
train_size = int(len(data) * 0.75)
train, test = torch.utils.data.random_split(data, [train_size, len(data) - train_size])

# Prepare data loaders
train_loader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False)

    
# Initialize model
model = MemoryPredict(windows_size=windows_size).cuda()
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    model.train()
    for inputs, label in train_loader:
        inputs, label = inputs.cuda(), label.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Batch Loss: {loss.item()}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, label in test_loader:
            inputs, label = inputs.cuda(), label.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            val_loss += loss.item()
    val_loss /= len(test_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")
    
# Testing
model.eval()
test_loss, correct, total = 0, 0, 0
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        test_loss += loss.item()
        for i in range(len(outputs)):
            total += 1
            if torch.argmax(outputs[i]) == inputs[i]:
                correct += 1

