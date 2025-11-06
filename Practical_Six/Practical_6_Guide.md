# Practical 6:

## Transformer Architecture Implementation (PyTorch)

**Course:** DAM202 [Year3-Sem1]

**Focus:** Implementation of the original "Attention Is All You Need" (Vaswani et al., 2017) Transformer architecture.

---

## Objective

The goal of this assignment is to gain a deep, practical understanding of the seminal **Transformer** architecture by implementing it from scratch using **PyTorch**. This exercise will reinforce core concepts such as Multi-Head Attention, Positional Encoding, and the complete Encoder-Decoder framework.

---

## Part 1: PyTorch Code Implementation (70% Weightage)

Students must implement the complete, original **Transformer architecture** using the PyTorch framework (`torch.nn`). The implementation must be **modular**, **well-documented**, and adhere to best practices.

### 1. Core Components Implementation

Implement the following components as distinct, well-documented PyTorch modules (`nn.Module`):

- **Scaled Dot-Product Attention:** Implement the core attention function, including the scaling factor $\frac{1}{\sqrt{d_k}}$.
- **Multi-Head Attention (MHA):** Implement the full mechanism: linear projections (Q, K, V), splitting into $h$ heads, attention calculation, concatenation, and final linear projection.
- **Position-wise Feed-Forward Network (FFN):** A two-linear-layer network with a ReLU activation between the layers.
- **Positional Encoding (PE):** Implement the **fixed sine and cosine** positional encodings, added to the input embeddings.
- **Encoder Layer:** A single block containing a Multi-Head **Self-Attention** sub-layer and an FFN sub-layer, each followed by a **Residual Connection** and **Layer Normalization**.
- **Decoder Layer:** A single block containing a Masked Multi-Head Self-Attention sub-layer, a Multi-Head **Cross-Attention** sub-layer (attending to Encoder output), and an FFN sub-layer, all with Residual Connections and Layer Normalization.
- **Transformer Model:** The complete model, stacking $N=6$ Encoder layers and $N=6$ Decoder layers, including input/output embedding layers and the final output projection.

### 2. Masking Requirements

Implement and apply the required masks:

- **Padding Mask:** Used in Self-Attention (Encoder and Decoder) and Cross-Attention to ignore `<PAD>` tokens.
- **Look-Ahead (Causal) Mask:** Applied within the Decoder's Self-Attention sub-layer to ensure causality (preventing attention to future tokens).

### 3. Standard Hyperparameters (Base Model Configuration)

Your implementation must adhere to the **Transformer (Base Model)** parameters from the paper to ensure dimensional correctness:

| Parameter                          | Notation      | Value    | Description                                     |
| :--------------------------------- | :------------ | :------- | :---------------------------------------------- |
| **Model Dimension**                | $d_{model}$   | **512**  | Size of embeddings and sub-layer outputs.       |
| **Number of Layers**               | $N$           | **6**    | Number of layers in Encoder and Decoder stacks. |
| **Number of Heads**                | $h$           | **8**    | Number of parallel attention heads.             |
| **Key/Value Dimension (per head)** | $d_k, d_v$    | **64**   | Calculated as $d_{model} / h$.                  |
| **Feed-Forward Inner Dim**         | $d_{ff}$      | **2048** | Inner dimension of the FFN.                     |
| **Dropout Rate**                   | $p_{dropout}$ | **0.1**  | Applied to sub-layer outputs and embeddings.    |

### 4. Basic Functionality Test

Provide a minimal Python script or notebook demonstrating:

- Instantiation of the complete `Transformer` model using the above hyperparameters.
- A forward pass with small, randomly generated dummy input tensors (for a sequence-to-sequence task) to verify all components are correctly connected and dimensions are valid.

---

## Part 2: Report and Documentation (30% Weightage)

Submit a formal, academic report (PDF format) accompanying your code.

### 1. Architectural Explanation

Provide a clear explanation of:

- The overall structure of the **Encoder** and **Decoder** stacks.
- The mathematical formulation and role of **Scaled Dot-Product Attention** and how **Multi-Head Attention** works conceptually.
- The function of **Positional Encoding** and the necessity of both the **Padding Mask** and the **Look-Ahead Mask**.

### 2. Code Structure and Design

- Detail the class hierarchy and modular design choices (e.g., why separate `MHA` from `EncoderLayer`).
- Explain the role and implementation of **Residual Connections** and **Layer Normalization** in your code.

### 3. Hand-Drawn Architecture

Create a **neat, original, hand-drawn diagram** that visually maps the theoretical Transformer architecture directly to the **specific Python classes/modules** implemented in your PyTorch code.

- **Labeling:** Clearly label blocks with your custom PyTorch class names (e.g., `MyMultiHeadAttention`, `MyEncoderLayer`).
- **Dimensions:** Annotate the diagram with the input/output tensor shapes (e.g., $(BatchSize, SeqLen, d_{model})$) at the boundaries of the major components.

---

## Submission

- **Code:** A single, well-commented Python file or an organized Jupyter Notebook (`.ipynb`).
- **Report:** A formal markdown document including all written explanations and an embedded image of the hand-drawn architecture.
- **Repository:** A formal Github Reposiroty
- **Link:** All are required to submit repo link while submitting the practical on College VLE.
