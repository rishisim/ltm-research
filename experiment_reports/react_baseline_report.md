# Experiment Report: ReAct Baseline (Gemini 2.5 Flash)

**Date**: 2025-12-31
**Model**: `gemini-2.5-flash`
**Environments**: 10 (Eval Out of Distribution)

## Summary
The baseline ReAct run achieved a **0% success rate (0/10)**.

## Comparison to Reflexion Paper
The user asked: *"Is this expected? Did the original paper do something different?"*

**Answer**: 
1.  **Prompting Strategy**: The Reflexion paper (Shinn et al., 2023) used **2-shot prompting** (just like we are). The few-shot examples were specifically chosen to guide GPT-3 to use the correct syntax.
2.  **Model Differences**: GPT-3.5-turbo closely adhered to the few-shot pattern `put X in/on Y`. Gemini 2.5 Flash is more compliant with natural language and often "corrects" this to `put X in Y`, which paradoxically causes the strict environment to fail.
3.  **No Hidden Code**: Our audit of `alfworld_trial.py` confirms there is no hidden post-processing or "magic" in the original code. It relies 100% on the model following the prompt's examples.

## Root Causes of Failure
1.  **Strict Syntax Mismatch** (`Major`):
    - **Prompt**: Uses `put object in/on receptacle` (required by Env).
    - **Gemini**: Outputs `put object in receptacle` (missing `/on`).
    - **Outcome**: Environment returns `Nothing happens.`
    
2.  **Thought formatting** (`Major`):
    - **Gemini**: Often splits thoughts across lines (e.g., `think: ... \n > action`).
    - **Code**: `llm()` function stops at `\n`, so it cuts off the action or sends the second half as an invalid command.

3.  **Logical Ordering** (`Minor`):
    - Agent occasionally attempts `put` without first navigating to the target (`go to`).

## Recommendation
This is **NOT a code bug** in our migration. It is a **prompt-model mismatch**.

To fix this, we should not change the code logic, but rather:
1.  **Add a System Instruction**: Explicitly tell Gemini to follow the `put X in/on Y` format.
2.  **Normalize Actions**: (Optional) Add a regex fix in `alfworld_trial.py` to handle `in` vs `in/on`.

We are ready to proceed to **Reflexion Implementation**, which is designed to inherently fix these kinds of "stuck" loops by letting the agent reflect on the `Nothing happens` feedback.
