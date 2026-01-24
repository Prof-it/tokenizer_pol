
class ModelConfig:
    vocab_size = 30_000
    d_model = 768
    num_heads = 12
    num_layers = 12
    context_size = 512
    d_ff = 4 * d_model
    dropout = 0.1
    learning_rate = 3e-4

class TrainingConfig:
    batch_size = 64
    num_workers = 4
    max_epochs = 4