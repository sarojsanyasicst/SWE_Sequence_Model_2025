# Multi-Task Learning Project Guide: NER and QA

**Course:** DAM202 [Year3-Sem1]

## **Project Goal**

Develop a single, multi-task Transformer-based model capable of concurrently performing **Named Entity Recognition (NER)** and **Question Answering (QA)**, utilizing a shared encoder architecture to leverage knowledge transfer between the tasks.

## **Prerequisites**

Students are expected to be proficient in:

- **Deep Learning Frameworks:** PyTorch (`torch`) or TensorFlow (`tf`).
- **Sequence Models:** Transformers and their architectures (BERT, RoBERTa, etc.).
- **Hugging Face Ecosystem:** `transformers` library (for models/tokenizers) and `datasets` library (for data handling).
- **Core NLP Concepts:** NER (sequence labeling, IOB format) and QA (Extractive QA, SQuAD format).

---

## **Part 1: Theoretical Foundation (Multi-Task Learning)**

### **1.1. Multi-Task Architecture: Hard Parameter Sharing**

We will employ the most common and effective strategy: **Hard Parameter Sharing**.

- **Shared Encoder:** The core component (e.g., BERT, RoBERTa) is shared across all tasks. This allows the model to learn a **universal and robust input representation** that benefits both NER and QA.
- **Task-Specific Heads:** Dedicated, lightweight layers are placed _on top_ of the shared encoder's output. These heads perform the final, task-specific prediction.

**Architectural Concept:**

```text

                                    [Input Text]
                                          |
                                          V

                        [Shared Transformer Encoder (e.g., BERT)]
                                          |
                          +---------------+---------------+
                          |                               |
                          V                               V
              [NER Head (Linear Layer)]       [QA Head (Two Linear Layers)]
                          |                               |
                          V                               V
              [NER Output (Tags)]             [QA Output (Start/End Span)]
```

### **1.2. The Multi-Task Loss Function**

The model is trained to optimize the weighted sum of the individual task losses.

$$\mathcal{L}_{MTL}(\theta) = \lambda_{NER} \cdot \mathcal{L}_{NER}(\theta) + \lambda_{QA} \cdot \mathcal{L}_{QA}(\theta)$$

| Component                         | Description                                                            | Typical Loss Function                                |
| :-------------------------------- | :--------------------------------------------------------------------- | :--------------------------------------------------- |
| **$\mathcal{L}_{NER}$**           | Loss for the sequence labeling task.                                   | **Cross-Entropy Loss** (or CRF Loss)                 |
| **$\mathcal{L}_{QA}$**            | Loss for predicting the answer span indices.                           | **Cross-Entropy Loss** (for start index + end index) |
| **$\lambda_{NER}, \lambda_{QA}$** | Hyperparameters to balance task importance (often initialized to 1.0). |                                                      |

---

## **Part 2: Practical Implementation Guide (Hugging Face / PyTorch Focus)**

### **2.1. Dataset Selection and Preparation**

| Task    | Recommended Dataset | Hugging Face Name | Format                                           |
| :------ | :------------------ | :---------------- | :----------------------------------------------- |
| **NER** | CoNLL-2003          | `conll2003`       | Token-level tags (IOB format)                    |
| **QA**  | SQuAD (v1.1 or v2)  | `squad`           | Context, Question, Answer Span (start/end index) |

**Key Preprocessing Steps:**

1.  **Tokenization:** Use a consistent `AutoTokenizer` (e.g., for `bert-base-uncased`).
2.  **NER Handling:** Apply standard methods for aligning token tags with sub-word tokens (e.g., using `-100` for subsequent sub-tokens to ignore loss).
3.  **QA Input:** Concatenate inputs as: `[CLS] question [SEP] context [SEP]`. Adjust answer span indices to the new tokenized input positions.
4.  **Unified DataLoader:** Create a custom `MultiTaskDataLoader` that **samples batches** from the NER dataset and the QA dataset, typically using a **Round-Robin** or **Stochastic Sampling** strategy.

### **2.2. Model Architecture Customization (PyTorch Example)**

You must define a custom model class, for example, by extending `transformers.modeling_outputs.PreTrainedModel`.

1.  **Shared Base Model:**

    ```python
    from transformers import AutoModel
    self.encoder = AutoModel.from_pretrained('bert-base-uncased')
    self.hidden_size = self.encoder.config.hidden_size
    ```

2.  **Task Heads:**

    ```python
    # NER Head (mapping hidden states to number of tags)
    self.ner_head = nn.Linear(self.hidden_size, num_ner_labels)

    # QA Head (two layers for start and end logits)
    self.qa_start_head = nn.Linear(self.hidden_size, 1)
    self.qa_end_head = nn.Linear(self.hidden_size, 1)
    ```

3.  **`MultiTaskModel.forward()` Logic:** The forward pass needs to branch based on an input `task_name` or `task_id` flag.

    ```python
    def forward(self, input_ids, attention_mask, task_name=None, labels=None):
        # 1. Shared Encoder Pass
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        if task_name == 'ner':
            # 2. NER Head
            ner_logits = self.ner_head(sequence_output)
            # Calculate and return NER loss/logits
        elif task_name == 'qa':
            # 2. QA Head
            start_logits = self.qa_start_head(sequence_output).squeeze(-1)
            end_logits = self.qa_end_head(sequence_output).squeeze(-1)
            # Calculate and return QA loss/logits
        # ...
    ```

### **2.3. Custom Training Loop**

Since the standard `transformers.Trainer` does not directly support MTL, a **custom training loop** is mandatory.

1.  **Initialization:** Instantiate the `MultiTaskModel`, `AdamW` optimizer, and learning rate scheduler.
2.  **Iteration:** Loop over epochs, drawing batches from the **custom `MultiTaskDataLoader`**.
3.  **Loss Calculation:**
    - Determine the task for the current batch.
    - Perform `model.forward(..., task_name=current_task)`.
    - Calculate the task-specific loss ($\mathcal{L}_{Task}$).
    - **Total Loss:** Apply the weighting: `L_Total = L_Task * lambda_Task`.
4.  **Optimization:** `L_Total.backward()`, `optimizer.step()`, `optimizer.zero_grad()`.

---

## **Part 3: Evaluation and Extensions**

### **3.1. Evaluation Metrics**

Model performance must be assessed for each task on its dedicated test set.

- **NER:** **F1-Score** (Micro-averaged over all entity types).
- **QA:** **F1-Score** (token overlap) and **Exact Match (EM)**.

### **3.2. Project Extensions (Advanced Topics)**

1.  **Loss Balancing:** Experiment with the loss weights ($\lambda_{NER}, \lambda_{QA}$). Consider implementing **GradNorm** for dynamic loss weighting.
2.  **Soft Parameter Sharing:** Modify the architecture to share only the bottom $N$ layers of the encoder and dedicate the top $M$ layers to each task to potentially mitigate negative interference.
3.  **Parameter-Efficient Fine-Tuning (PEFT):** Incorporate methods like **LoRA** (Low-Rank Adaptation) for task-specific adaptations, reducing the number of task-specific parameters required.
