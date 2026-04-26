# Tool Retrieval Methodology

## Methodology: Hybrid Tool Retrieval Mechanism (Help Tool)

Complementing the autonomous context retrieval, we implemented an on-demand **Tool Retrieval** mechanism (exposed as a "Help Tool") that allows the agent to actively query the Memory Bank when it encounters a specific obstacle. This system matches the agent's current issue description with historical issues to provide targeted solutions using a **hybrid retrieval approach** that combines dense semantic search with sparse lexical matching.

### 1. Query Processing
When the agent invokes the help tool with an `issue` description (e.g., "cannot find apple 1"), the system first preprocesses the query:
*   **Digit Removal**: Digits are stripped from the query to generalize the search to object types and actions rather than specific instance IDs.
*   **Normalization**: Whitespace is collapsed and text is lowercased.

### 2. Hybrid Retrieval Architecture
The system employs a two-stage retrieval process to capture both semantic meaning and exact keyword matches.

#### A. Dense Retrieval (Semantic)
*   **Embedding**: The cleaned query is embedded into a high-dimensional vector $v_q$ using Google's `gemini-embedding-001` model.
*   **Similarity**: We compute the **Cosine Similarity** between $v_q$ and the pre-computed embeddings of all `issue_text` fields in the Knowledge Base. This captures the semantic intent of the problem.

#### B. Sparse Retrieval (Lexical)
*   **Index Construction**: A BM25 index is constructed over the Knowledge Base. Each document in the index is a concatenation of the entry's `obj_type`, `verbs`, `issue_text`, and `learning_text`.
*   **Scoring**: The system calculates the **BM25 score** for the tokenized query against the index, prioritizing entries with exact keyword matches.

### 3. Candidate Selection and Normalization
To efficiently rank the results, we select a candidate pool consisting of the union of:
*   Top 100 results from Dense Retrieval
*   Top 100 results from Sparse Retrieval

For this candidate set, we normalize the scores to a common scale $[0, 1]$:
*   **Cosine Normalization**: Clamped to $[0, 1]$.
*   **BM25 Normalization**: Min-max normalization based on the scores within the candidate set:
    $$ \text{BM25}_{norm} = \frac{\text{BM25} - \text{BM25}_{min}}{\text{BM25}_{max} - \text{BM25}_{min}} $$

### 4. Multiplicative Scoring & Ranking
The final ranking is determined by a **Multiplicative Scoring** function that fuses the hybrid retrieval scores and incorporates the reliability (validation level) of the historical entry.

*   **Hybrid Score Fusion**: We combine the normalized scores with a weighted sum, prioritizing semantic similarity:
    $$ S_{hybrid} = 0.7 \times \text{Cosine}_{norm} + 0.3 \times \text{BM25}_{norm} $$

*   **Validation Weights ($\omega$)**: We assign scalar weights based on the entry's `valid_level`:
    *   `VALID_NEXT_TRIAL`: $\omega = 1.0$ (High confidence)
    *   `VALID_SAME_TRIAL`: $\omega = 0.9$ (Medium confidence)
    *   `CANDIDATE`: $\omega = 0.6$ (Lower confidence)

*   **Final Scoring Formula**:
    $$ \text{Score}(e) = S_{hybrid} \times \omega_{e.\text{validation}} $$

### 5. Output Generation
*   **Ranking**: Candidates are ranked by the final score in descending order.
*   **Selection**: The top $k$ (default $k=3$) entries are selected.
*   **Presentation**: The system returns a structured list mapping the matched "Similar Issue" to its corresponding "Solution" (learning), providing the agent with actionable advice for its specific blocker.

---

## Algorithm Pseudocode

**Algorithm 2:** Hybrid Tool Retrieval

**Input:**
*   $I_q$: Agent's current issue description
*   $\mathcal{K}$: Knowledge Base
*   $k$: Number of results (default 3)

**Parameters:**
*   $\alpha = 0.7$ (Dense weight), $\beta = 0.3$ (Sparse weight)
*   $W = \{ \text{NEXT\_TRIAL}: 1.0, \text{SAME\_TRIAL}: 0.9, \text{CANDIDATE}: 0.6 \}$

**Procedure:**
1.  **// Preprocess**
    $q' \leftarrow \text{RemoveDigits}(I_q)$
    
2.  **// Hybrid Retrieval**
    $v_q \leftarrow \text{Embed}(q')$
    $Scores_{cos} \leftarrow \text{CosineSimilarity}(v_q, \mathcal{K}.\text{embeddings})$
    $Scores_{bm25} \leftarrow \text{BM25}(q', \mathcal{K}.\text{index})$
    
3.  **// Candidate Selection**
    $C \leftarrow \text{Top}(Scores_{cos}, 100) \cup \text{Top}(Scores_{bm25}, 100)$
    
4.  **// Score Fusion**
    $\mathcal{S} \leftarrow \emptyset$
    **for each** $i \in C$ **do**
        $norm_{cos} \leftarrow \text{Clamp}(Scores_{cos}[i], 0, 1)$
        $norm_{bm25} \leftarrow \text{MinMax}(Scores_{bm25}[i], C)$
        
        $S_{hybrid} \leftarrow \alpha \cdot norm_{cos} + \beta \cdot norm_{bm25}$
        $w \leftarrow W[\mathcal{K}[i].\text{valid\_level}]$
        
        $FinalScore \leftarrow S_{hybrid} \times w$
        Add $(\mathcal{K}[i], FinalScore)$ to $\mathcal{S}$
    **end for**
    
5.  **// Rank and Return**
    Sort $\mathcal{S}$ by $FinalScore$ descending
    **return** first $k$ elements of $\mathcal{S}$
