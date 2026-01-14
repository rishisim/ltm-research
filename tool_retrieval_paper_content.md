# Tool Retrieval Methodology

## Methodology: Tool Retrieval Mechanism (Help Tool)

Complementing the autonomous context retrieval, we implemented an on-demand **Tool Retrieval** mechanism (exposed as a "Help Tool") that allows the agent to actively query the Memory Bank when it encounters a specific obstacle. This system matches the agent's current issue description with historical issues to provide targeted solutions.

### 1. Embedding and Similarity Search
Similar to context retrieval, the tool retrieval process relies on semantic vector space modeling to find relevant past experiences.
*   **Query Embedding**: The agent provides a natural language description of its current `issue` (e.g., "cannot find the apple"). This query is embedded into a vector $v_q$ using the `text-embedding-004` model.
*   **Issue Matching**: The system computes the **Cosine Similarity** between $v_q$ and the pre-computed embeddings of all `issue_text` fields in the Knowledge Base. This focuses the search specifically on the *problem* encountered, rather than the general task description.

### 2. Multiplicative Scoring System
To ensure the advice provided is both relevant and reliable, we employ a **Multiplicative Scoring** function that combines semantic similarity with the validation confidence of the historical entry.
*   **Validation Weights ($\omega$)**: We assign scalar weights to entries based on their `valid_level`:
    *   `VALID_NEXT_TRIAL` & `VALID_SAME_TRIAL`: $\omega = 1.0$ (High confidence)
    *   `CANDIDATE`: $\omega = 0.5$ (Lower confidence)
    *   *Note: This effectively penalizes unvalidated hypotheses, requiring them to be twice as semantically similar as a validated entry to achieve the same rank.*
*   **Scoring Formula**:
    $$ \text{Score}(e) = \text{Sim}(v_q, v_{e.\text{issue}}) \times \omega_{e.\text{validation}} $$

### 3. Ranking and Selection
*   **Ranking**: All entries in the Knowledge Base are ranked in descending order based on their calculated Multiplicative Score.
*   **Top-k Selection**: The top $k$ (default $k=3$) entries are selected.
*   **Presentation**: The system returns a structured list containing the similar `issue` and its corresponding `learning`. This direct mapping of "Similar Issue" $\rightarrow$ "Solution" provides the agent with actionable steps to resolve its immediate blocker.

---

## Algorithm Pseudocode

**Algorithm 2:** Tool Retrieval (Help Tool)

**Input:**
*   $I_q$: Agent's current issue description (query)
*   $\mathcal{K}$: Knowledge Base containing historical tasks
*   $k$: Number of results to return (e.g., $k=3$)

**Output:**
*   $R$: List of top-k relevant solutions

**Parameters:**
*   $W$: Map of validation weights (e.g., $\{ \text{VALIDated}: 1.0, \text{CANDIDATE}: 0.5 \}$)

**Procedure:**

1.  **// Embed Query Issue**
    $v_q \leftarrow \text{Embed}(I_q)$

2.  **// Score All Entries**
    $\mathcal{S} \leftarrow \emptyset$
    **for each** entry $e \in \mathcal{K}$ **do**
        $v_e \leftarrow \text{GetCachedEmbedding}(e.\text{issue\_text})$
        
        **// Calculate Semantic Similarity**
        $sim \leftarrow \text{CosineSimilarity}(v_q, v_e)$
        
        **// Apply Validation Weight**
        $w \leftarrow W[e.\text{valid\_level}]$
        $score \leftarrow sim \times w$
        
        Add $(e, score)$ to $\mathcal{S}$
    **end for**

3.  **// Rank and Select**
    Sort $\mathcal{S}$ by $score$ descending
    $\mathcal{R}_{top} \leftarrow \text{First } k \text{ elements of } \mathcal{S}$

4.  **// Format Output**
    $R \leftarrow \emptyset$
    **for each** $(e, score)$ in $\mathcal{R}_{top}$ **do**
        $r \leftarrow \{ \text{Issue}: e.\text{issue\_text}, \text{Learning}: e.\text{learning\_text}, \text{Score}: score \}$
        Add $r$ to $R$
    **end for**

    **return** $R$
