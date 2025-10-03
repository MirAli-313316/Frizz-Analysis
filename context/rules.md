# Context Engineering Rules for [Your App Name]

## Purpose
[Describe your app's main purpose here]

## Core Principles

1. **Clarity Over Cleverness**: Write code that's easy to understand
2. **Explicit Context**: Always provide clear context to models
3. **Fail Gracefully**: Handle model failures and edge cases
4. **Iterative Refinement**: Start simple, enhance based on results

## Model Interaction Patterns

### When calling Hugging Face models:
- Always specify max_tokens/max_length
- Include temperature/top_p parameters explicitly
- Log model inputs and outputs for debugging
- Implement retry logic with exponential backoff

### Context Window Management:
- Track token counts for inputs
- Implement context truncation strategies
- Preserve most relevant information when truncating

## Development Workflow

1. Define the feature clearly
2. Create example inputs/outputs
3. Implement basic version
4. Test with edge cases
5. Refine based on results