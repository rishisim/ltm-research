# Context Retrieval Methodology

## Methodology: Context Retrieval Mechanism

To enable the agent to leverage past experiences effectively, we implemented a **Context Retrieval** module that dynamically selects the most relevant learnings from the Memory Bank. This process ensures that the agent receives targeted guidance based on similar past tasks, facilitating generalization to unseen scenarios.

### 1. Embedding and Similarity Search
The core of the retrieval mechanism utilizes semantic similarity to identify relevant historical tasks.
*   **Embedding Model**: We employ the `gemini-embedding-001` model (Google GenAI) to generate high-dimensional vector representations of task descriptions.
*   **Vector Space**: Both the current task description query $q$ and all historical task descriptions in the Knowledge Base $\{d_1, d_2, ..., d_n\}$ are mapped to a shared vector space.
*   **Similarity Metric**: We compute the **Cosine Similarity** between the query vector $v_q$ and each historical task vector $v_{d_i}$ to quantify relevance:
    $$ \text{Sim}(v_q, v_{d_i}) = \frac{v_q \cdot v_{d_i}}{\|v_q\| \|v_{d_i}\|} $$
*   **Initial Retrieval**: The system identifies the top-$k$ ($k=5$) most similar tasks from the `mem_learning_counts` index, which tracks distinct tasks and the quantity of learnings associated with them.

### 2. Dynamic Learning Selection Strategy
Instead of retrieving a fixed number of learnings, we implemented a **dynamic selection strategy** ("learning count selection") to adapt to the complexity of the task.
*   **Adaptive Threshold**: The system calculates the maximum number of learnings associated with any of the top-5 retrieved tasks ($L_{max}$).
*   **Retrieval Quota**: The final number of learnings to be presented to the agent, $N_{pick}$, is dynamically set as:
    $$ N_{pick} = \lceil 1.5 \times L_{max} \rceil $$
    This scaling factor ensures that the agent receives sufficient context—covering not just the single best match, but a broader range of potentially useful insights from neighboring tasks.

### 3. Prioritization and Re-ranking
Once the pool of potential learnings (associated with the top-$k$ tasks) is retrieved, a rigorous re-ranking process is applied to prioritize high-quality, validated knowledge over unverified hypotheses.
*   **Validation Hierarchy**: Learnings are sorted primarily by their validation status in the following descending order of priority:
    1.  **`VALID_NEXT_TRIAL`**: Learnings that were successfully applied and verified in a subsequent trial (highest confidence).
    2.  **`VALID_SAME_TRIAL`**: Learnings that led to immediate success within the same trial.
    3.  **`CANDIDATE`**: Hypotheses generated but not yet strictly validated (lowest confidence).
*   **Secondary Sorting**: Within the same validation tier, learnings are ordered by the semantic **similarity score** of their source task to the current query.
*   **Filtering**: To prevent data leakage during self-evaluation or redundancy, exact matches (similarity score $\ge 1.0$) are filtered out.

### 4. Final Context Construction
The top $N_{pick}$ learnings after re-ranking are selected and formatted into a structured prompt component. This provides the Large Language Model (LLM) with a curated list of "Relevant Learnings," explicitly detailing the past **Issue** (the mistake made) and the corresponding **Learning** (the correction), guiding the agent to avoid repeating specific errors.

---

## Algorithm Pseudocode

**Algorithm 1:** Context Retrieval from Memory Bank

**Input:**
*   $q$: Current task description (query)
*   $\mathcal{K}$: Knowledge Base containing historical tasks $\mathcal{T}$ and learnings $\mathcal{L}$
*   $k$: Number of nearest neighbor tasks to retrieve (e.g., $k=5$)

**Output:**
*   $C$: Selected set of relevant learnings (context)

**Parameters:**
*   $\alpha$: Scaling factor for context window size (e.g., $\alpha = 1.5$)
*   $V$: Validation priority map mapping status to integer rank (where `VALID_NEXT_TRIAL` > `VALID_SAME_TRIAL` > `CANDIDATE`)

**Procedure:**

1.  **// Embed Query**
    $v_q \leftarrow \text{Embed}(q)$

2.  **// Retrieve Top-k Similar Tasks**
    $\mathcal{S} \leftarrow \emptyset$
    **for each** unique task $t \in \mathcal{K}$ **do**
        $v_t \leftarrow \text{Embed}(t.\text{description})$
        $s_t \leftarrow \text{CosineSimilarity}(v_q, v_t)$
        Add $(t, s_t)$ to $\mathcal{S}$
    **end for**
    $\mathcal{T}_{top} \leftarrow \text{Top-k tasks in } \mathcal{S} \text{ sorted by } s_t \text{ descending}$

3.  **// Determine Dynamic Context Size**
    $N_{counts} \leftarrow \{ \text{CountLearnings}(t) \mid t \in \mathcal{T}_{top} \}$
    $L_{max} \leftarrow \max(N_{counts})$
    $N_{pick} \leftarrow \lceil \alpha \times L_{max} \rceil$

4.  **// Gather Candidate Learnings**
    $\mathcal{L}_{pool} \leftarrow \emptyset$
    **for each** $(t, s_t) \in \mathcal{T}_{top}$ **do**
        **for each** learning $l \in \text{GetLearnings}(t)$ **do**
            $l.similarity \leftarrow s_t$
            Add $l$ to $\mathcal{L}_{pool}$
        **end for**
    **end for**

5.  **// Re-rank by Validation Status then Similarity**
    **function** $\text{Rank}(l)$:
        **return** $(V[l.valid\_level], l.similarity)$
    
    Sort $\mathcal{L}_{pool}$ descending using $\text{Rank}(\cdot)$

6.  **// Select Final Context**
    $C \leftarrow \text{First } N_{pick} \text{ elements of } \mathcal{L}_{pool}$

    **return** $C$
