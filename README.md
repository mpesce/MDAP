# MDAP: Massively Decomposed Agentic Processes

Implementation of the MAKER system from the paper:

**"Solving a Million-Step LLM Task with Zero Errors"**
*Meyerson et al., Cognizant AI Lab, 2025*
arXiv:2511.09030v1

## Overview

This repository implements **MAKER** (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging), a system that successfully solves tasks requiring over **1 million LLM steps with zero errors**.

### Key Results from the Paper

- ✅ **First system** to solve a task with 1,048,575 LLM steps (20-disk Towers of Hanoi) with zero errors
- 📊 **Cost scales as Θ(s ln s)** with proper decomposition (vs exponential without it)
- 🎯 **Small models suffice**: Uses gpt-4.1-mini effectively (non-reasoning models work well)
- 🔬 **Theoretical foundation**: Provides scaling laws for cost and reliability

### Three Core Components

1. **Maximal Agentic Decomposition (MAD)**: Break tasks into minimal subtasks - each agent handles exactly **one step**

2. **First-to-ahead-by-k Voting**: Multiple agents vote on each step; continue until one answer is k votes ahead of all others (based on Sequential Probability Ratio Testing)

3. **Red-flagging**: Discard responses showing signs of unreliability:
   - Overly long responses (>750 tokens)
   - Incorrectly formatted responses

## Installation

```bash
# Clone the repository
git clone https://github.com/mpesce/MDAP.git
cd MDAP

# Install dependencies (Python 3.8+)
pip install openai  # If using OpenAI
# or install your preferred LLM library
```

## Quick Start

### Basic Example

```python
from mdap.core.agent import TowersOfHanoiAgent
from mdap.core.maker import MAKER
from mdap.domains.towers_of_hanoi import create_initial_state, get_optimal_num_steps

# 1. Create an LLM function (example with OpenAI)
def llm_fn(system_prompt, user_prompt, kwargs):
    from openai import OpenAI
    client = OpenAI(api_key="your-api-key")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=kwargs.get("max_tokens", 750),
        temperature=kwargs.get("temperature", 0.1)
    )
    return response.choices[0].message.content

# 2. Create agent
agent = TowersOfHanoiAgent(
    llm_fn=llm_fn,
    n_disks=3,  # Start small
    max_tokens=750,
    temperature=0.1
)

# 3. Create MAKER system
maker = MAKER(
    agent=agent,
    k=2,  # Vote margin (paper uses k=3 for 20-disk problem)
    voting_method="first-to-ahead-by-k",
    verbose=True
)

# 4. Solve
initial_state = create_initial_state(3)
num_steps = get_optimal_num_steps(3)  # 2^3 - 1 = 7

solution = maker.generate_solution(
    initial_state=initial_state,
    num_steps=num_steps
)

print(f"Solved with {len(solution)} steps!")
```

### Run the Example

```bash
# With mock LLM (for testing)
python mdap/experiments/example_towers.py

# With real LLM (set API key first)
export OPENAI_API_KEY="your-key"
# Edit example_towers.py to uncomment OpenAI section
python mdap/experiments/example_towers.py
```

## Architecture

```
mdap/
├── core/
│   ├── agent.py          # LLM agent with MAD (m=1 steps)
│   ├── voting.py         # First-to-ahead-by-k voting (Algorithm 2)
│   ├── parser.py         # Red-flagging parser (Section 3.3)
│   └── maker.py          # Main MAKER system (Algorithm 1)
├── domains/
│   └── towers_of_hanoi.py  # ToH specific utilities
├── analysis/
│   └── (future: scaling law analysis)
└── experiments/
    └── example_towers.py   # Demonstration script
```

## How It Works

### Algorithm Overview

MAKER implements Algorithm 1 from the paper:

```python
def generate_solution(initial_state, num_steps):
    action_list = []
    state = initial_state
    prev_move = initial_move

    for step in range(num_steps):
        # Algorithm 2: First-to-ahead-by-k voting
        move, state = do_voting(state, prev_move, k)
        action_list.append(move)
        prev_move = move

    return action_list
```

### Voting Mechanism (Algorithm 2)

```python
def do_voting(state, prev_move, k):
    vote_counts = {}

    while True:
        # Algorithm 3: Get a vote (with red-flagging)
        try:
            vote = agent.get_vote(state, prev_move)
            vote_counts[vote] += 1

            # Check if winner (first-to-ahead-by-k)
            if vote_counts[vote] >= k + max(other_counts):
                return vote
        except ValueError:
            # Red-flagged response, discard and continue
            continue
```

## Scaling Laws

The paper derives theoretical scaling laws for MAKER:

### Cost Scaling
With maximal decomposition (m=1):

```
E[cost] = Θ(s ln s)
```

Where:
- s = number of steps
- Cost grows **log-linearly** with steps (efficient!)

### Vote Margin Required
For probability of success t:

```
k_min = ⌈ln(t^(-m/s) - 1) / ln((1-p)/p)⌉ = Θ(ln s)
```

### Key Insight
Without decomposition (large m), cost grows **exponentially**. With maximal decomposition (m=1), cost grows **log-linearly**.

## Customization

### Using Different LLM Providers

```python
# Anthropic Claude
def claude_llm(system_prompt, user_prompt, kwargs):
    from anthropic import Anthropic
    client = Anthropic(api_key="your-key")

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        max_tokens=kwargs.get("max_tokens", 750),
        temperature=kwargs.get("temperature", 0.1)
    )
    return response.content[0].text

# Open-source via together.ai, ollama, etc.
def together_llm(system_prompt, user_prompt, kwargs):
    import together
    together.api_key = "your-key"

    response = together.Complete.create(
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        prompt=f"{system_prompt}\n\n{user_prompt}",
        max_tokens=kwargs.get("max_tokens", 750),
        temperature=kwargs.get("temperature", 0.1)
    )
    return response['output']['choices'][0]['text']
```

### Tuning Parameters

```python
# For more reliable results (higher cost)
maker = MAKER(agent, k=5, voting_method="first-to-ahead-by-k")

# For faster results (lower reliability)
maker = MAKER(agent, k=2, voting_method="first-to-k")

# Adjust agent parameters
agent = TowersOfHanoiAgent(
    llm_fn=llm_fn,
    n_disks=20,
    max_tokens=500,      # Lower threshold for red-flagging
    temperature=0.0      # Deterministic (for first vote)
)
```

## Reproducing Paper Results

To reproduce the million-step experiment from the paper:

```python
# Configuration from Section 4.4
agent = TowersOfHanoiAgent(
    llm_fn=gpt_4_1_mini_llm,  # Paper uses gpt-4.1-mini
    n_disks=20,                # 2^20 - 1 = 1,048,575 steps
    max_tokens=750,            # Red-flag threshold
    temperature=0.1            # After first vote (0.0 for first)
)

maker = MAKER(
    agent=agent,
    k=3,                       # Paper uses k=3
    voting_method="first-to-ahead-by-k",
    verbose=True
)

# This will take several hours and cost ~$3.5K with gpt-4.1-mini
solution = maker.generate_solution(
    initial_state=create_initial_state(20),
    num_steps=get_optimal_num_steps(20)
)
```

**Expected results** (from paper):
- Total LLM calls: ~3-4 million
- Valid votes: ~99.8%
- Red-flagged: ~0.2%
- Solution: ✅ Zero errors across 1,048,575 steps

## Key Findings from the Paper

1. **Small models work**: gpt-4.1-mini (non-reasoning) outperforms larger reasoning models in cost-effectiveness

2. **Decomposition is critical**: Cost grows exponentially without decomposition, log-linearly with MAD

3. **Red-flagging helps**: Discarding long/malformed responses reduces correlated errors significantly

4. **Voting converges fast**: Most steps require only k+1 to k+3 votes (exponential tail)

5. **Error decorrelation**: Independent sampling + red-flagging achieves near-i.i.d. error rates

## Citation

```bibtex
@article{meyerson2025solving,
  title={Solving a Million-Step LLM Task with Zero Errors},
  author={Meyerson, Elliot and Paolo, Giuseppe and Dailey, Roberto and
          Shahrzad, Hormoz and Francon, Olivier and Hayes, Conor F and
          Qiu, Xin and Hodjat, Babak and Miikkulainen, Risto},
  journal={arXiv preprint arXiv:2511.09030},
  year={2025}
}
```

## Future Work

The paper discusses several extensions:

1. **Recursive decomposition** (Appendix F): Automatically decompose arbitrary tasks
2. **Multiple agent types**: Decomposition agents, discriminators, solvers
3. **Better error decorrelation**: Prompt paraphrasing, diverse models
4. **General task domains**: Beyond Towers of Hanoi to real-world problems

## License

This implementation is provided for research and educational purposes.

## Contact

For questions about this implementation:
- Open an issue on GitHub
- Email: mpesce@gmail.com

For questions about the paper:
- Contact: elliot.meyerson@cognizant.com

---

**Built with 🧠 by following the MDAP framework**
