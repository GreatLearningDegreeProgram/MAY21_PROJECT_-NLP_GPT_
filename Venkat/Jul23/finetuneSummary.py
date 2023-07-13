from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
from torch.optim import Adam

class SummarizationDataset(Dataset):
    def __init__(self, source_docs, summaries):
        self.source_docs = source_docs
        self.summaries = summaries

    def __len__(self):
        return len(self.source_docs)

    def __getitem__(self, idx):
        return self.source_docs[idx], self.summaries[idx]

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def encode_text(text):
    return tokenizer.encode(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

model = GPT2LMHeadModel.from_pretrained('gpt2')

batch_size = 8
num_epochs = 3
learning_rate = 1e-5

train_dataset = SummarizationDataset(train_source_docs, train_summaries)
valid_dataset = SummarizationDataset(valid_source_docs, valid_summaries)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for source_docs, summaries in train_dataloader:
        optimizer.zero_grad()
        source_docs = source_docs.to(device)
        summaries = summaries.to(device)

        # Forward pass
        outputs = model(input_ids=source_docs, labels=summaries)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Calculate average training loss
    avg_train_loss = total_loss / len(train_dataloader)

    # Evaluation loop
    model.eval()
    total_valid_loss = 0
    with torch.no_grad():
        for source_docs, summaries in valid_dataloader:
            source_docs = source_docs.to(device)
            summaries = summaries.to(device)

            # Forward pass
            outputs = model(input_ids=source_docs, labels=summaries)
            loss = outputs.loss
            total_valid_loss += loss.item()

    # Calculate average validation loss
    avg_valid_loss = total_valid_loss / len(valid_dataloader)

    # Print training and validation loss for each epoch
    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}')

# Generate a summary for a new input text
def generate_summary(input_text):
    input_ids = encode_text(input_text)
    input_ids = input_ids.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0])
    return summary