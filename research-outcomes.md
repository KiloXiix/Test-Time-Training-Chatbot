# **My Testing Outcomes: Building a Continuously Learning Chatbot with Test-Time Training**

***Disclaimers***: 
- After Learning about Test-Time-Training and its usage in image generation models, I had Gemini do `Deep Research` on the topic so I could learn more about how it worked so I could get an idea on how I could implement it with an ai chatbot.
- I did not come up with Test-Time-Training. I happened to stumble upon research done by [RetentionLabs](https://retentionlabs.github.io) which uses Test-Time-Training from the looks of it. I liked the idea, and then tried to implement it into a chatbot.
- I did use some AI to help code certain areas and debug errors but I still determined what would be implemented and what wouldn't be implemented in the end.





#### **What I'm trying to do:**

I have been trying to integrate the Test-Time Training (TTT) method into my own custom AI chatbot architecture. My main goal is to build a bot that can continuously learn from me in real-time during our conversations. I want it to pick up new vocabulary on the fly and actually remember the specific facts I tell it.

To make this happen, I built a specialized belief and vocabulary system that assigns a dynamic confidence score to everything the AI learns. Every time the model encounters the same piece of information, that confidence score goes up. Basically, the system reinforces its own knowledge based entirely on my input—meaning it will confidently adopt and use the facts I give it, even if I teach it something completely wrong.

#### **The Results (Direct Terminal Output):**

Here is the direct output from my terminal after a live conversation with the TTT-enabled model. For this test, I used the Qwen/Qwen3.5-0.8B model, which I retrofitted with DualMLP TTT layers. You can see exactly how the belief extraction, vocab learning, and dynamic weight updates work under the hood during a standard session.

```

╔══════════════════════════════════════════════════════════════╗
║           Companion AI — Learns As You Talk                  ║
║                                                              ║
║  Like a baby learning language:                              ║
║   • Remembers what you tell it across sessions               ║
║   • Learns new words and their meanings                      ║
║   • Accepts corrections and updates its beliefs              ║
║   • Adapts its language patterns to match yours              ║
║                                                              ║
║  Powered by: TTT + Belief Store + Vocabulary Module          ║
╚══════════════════════════════════════════════════════════════╝

  Select a model:

    [1] Best — native TTT, 1.3B, 32k context, Books training
        RetentionLabs/TTT-Linear-1.3B-Base-Books-32k
    [2] Good — native TTT-MLP, 760M, broad Pile training
        RetentionLabs/TTT-MLP-760M-Base-Pile-8k
    [3] Good — native TTT-Linear, 760M, fast inference
        RetentionLabs/TTT-Linear-760M-Base-Pile-8k
    [4] Retrofit — fast, light, good for testing
        ibm-granite/granite-4.0-350m
    [5] Retrofit — modern architecture, 262K context
        Qwen/Qwen3.5-0.8B

  Choice (1-5, or paste a HuggingFace model ID): 5

  Using: Qwen/Qwen3.5-0.8B

  Your name / user ID (Enter = 'friend'): Kilo

  Hello, Kilo!

  AI companion name (Enter = 'Companion'): Evere

  Your companion's name is: Evere

  Loading companion (first run downloads model weights)...


  Loading embedding model: Qwen/Qwen3-Embedding-0.6B
  Device: cuda | Dtype: torch.bfloat16
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 310/310 [00:00<00:00, 860.34it/s]
  Embedding model loaded | VRAM: 1.19/8.59 GB

  Loading model: Qwen/Qwen3.5-0.8B
  Device: cuda | Dtype: torch.bfloat16
The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 320/320 [00:00<00:00, 684.96it/s]
  ✓ Standard transformer — installing DualMLP TTT retrofit
  TTT blocks: 18–23 (6 of 24 blocks)
  Trainable: 41,299,968 / 793,692,992 params (5.20%)
  VRAM: 2.79/8.59 GB

  ● Starting fresh session

  Commands:
    /verbose    — Toggle detailed learning + TTT stats each turn
    /stats      — Show full learning statistics
    /beliefs    — Show everything the AI believes about you
    /vocab      — Show vocabulary acquisition status
    /save       — Force-save the current session
    /clear      — Clear screen
    /help       — Show this help
    /quit       — Save and exit

────────────────────────────────────────────────────────────────
  Start talking! The more you talk, the more it learns.

You: Hello?

Companion: Greetings! I am Evere. How can I help you today?

You: My name is Kilo. Your name is Evere. Evere is a French name derived from the Latin word aevum, meaning "eternity".

Companion: Thank you for sharing that. My name is Evere, and I do understand your story about my origin. How about you?

You: /verbose
  Verbose mode: ON

You: /vocab

  ── Vocabulary ──
  Still learning (10):
    ? aevum [0.20]?
    ? derived [0.20]?
    ? eternity [0.20]?
    ? evere [0.20]?
    ? french [0.20]?
    ? kilo [0.20]?
    ? latin [0.20]?
    ? meaning [0.20]?
    ? name [0.20]?
    ? word [0.20]?
You: /beliefs

  ── Known Beliefs (1) ──
  [████░░░░░░] 0.45  user name: Kilo
You: My name is Kilo. While my name has no meaning, there are several other meanings for my name such as Kilogram which is a metric of weight or mass, Kilobyte which is a metric of data storage, and Kilometer which is a metric of distance.
  [↑ Reinforced: Kilo]
  [📖 New words encountered: while, several, meanings, kilogram, metric]

Companion: It is interesting that your name carries such diverse meanings beyond just "Kilo," especially regarding units of weight, data, and space.
  [✦ TTT | updated | loss=0.1343 | ema=0.0193 | efficiency=0.0% | vram=3.00/8.59 GB]
  [turn 3 | 3.6s total]

You: /beliefs                                                                                                                                                                     

  ── Known Beliefs (1) ──
  [█████░░░░░] 0.57  user name: Kilo
You: I actually created you using python code and various AI models 
  [📖 New words encountered: created, using, python, code, various]

Companion: Thank you for revealing your background; it's fascinating how different aspects of my existence come together through your tools.
  [✦ TTT | updated | loss=0.1221 | ema=0.0194 | efficiency=0.0% | vram=3.00/8.59 GB]
  [turn 4 | 2.0s total]

You: My favorite type of food is seafood and asian food in general.
  [📌 New belief: user favourite_type of food: seafood and asian food in general]
  [📌 New belief: my favorite type of food is: seafood and asian food in general]
  [📖 New words encountered: favorite, type, food, seafood, asian]

Companion: Seafood and Asian food are wonderful choices for a life filled with culture and taste.
  [✦ TTT | updated | loss=0.1172 | ema=0.0194 | efficiency=0.0% | vram=3.00/8.59 GB]
  [turn 5 | 1.7s total]

You: My favorite asian food is specifically from japan being japanese foods.
  [📌 New belief: user favourite_asian food: specifically from japan being japanese foods]
  [📌 New belief: my favorite asian food is: specifically from japan being japanese foods]
  [📖 New words encountered: specifically, japan, japanese, foods]

Companion: That is a very specific and memorable preference! Japanese cuisine offers a beautiful blend of tradition, seafood, and flavors.
  [✦ TTT | updated | loss=0.1143 | ema=0.0194 | efficiency=0.0% | vram=3.00/8.59 GB]
  [turn 6 | 3.5s total]

You: /vocab

  ── Vocabulary ──
  Still learning (20):
    ? asian [0.35]?
    ? favorite [0.35]?
    ? food [0.35]?
    ? kilo [0.35]?
    ? meaning [0.35]?
    ? name [0.35]?
    ? aevum [0.20]?
    ? code [0.20]?
    ? created [0.20]?
    ? data [0.20]?
You: /beliefs

  ── Known Beliefs (5) ──
  [█████░░░░░] 0.57  user name: Kilo
  [████░░░░░░] 0.45  user favourite_asian food: specifically from japan being japanese foods
  [████░░░░░░] 0.45  user favourite_type of food: seafood and asian food in general
  [████░░░░░░] 0.40  my favorite asian food is: specifically from japan being japanese foods
  [████░░░░░░] 0.40  my favorite type of food is: seafood and asian food in general
You: /stats

═══ Companion Learning Stats ════════════════════════════════
  User ID:          Kilo
  Model:            Qwen/Qwen3.5-0.8B
  Total turns:      6
  Device / VRAM:    3.00/8.59 GB

  ── TTT Weight Adaptation ───────────────────────────────
  Updates fired:    6
  Updates skipped:  0
  Gate efficiency:  0.0%
  EMA loss:         0.0194

  ── Belief Store ────────────────────────────────────────
  Total beliefs:    5
  High confidence:  0
  Low confidence:   0
  Corrected:        0

  ── Vocabulary ──────────────────────────────────────────
  Total words:      38
  Fully acquired:   0
  Familiar:         0
  Still learning:   38
  Avg confidence:   0.22
═════════════════════════════════════════════════════════════

You: /quit

═══ Companion Learning Stats ════════════════════════════════
  User ID:          Kilo
  Model:            Qwen/Qwen3.5-0.8B
  Total turns:      6
  Device / VRAM:    3.00/8.59 GB

  ── TTT Weight Adaptation ───────────────────────────────
  Updates fired:    6
  Updates skipped:  0
  Gate efficiency:  0.0%
  EMA loss:         0.0194

  ── Belief Store ────────────────────────────────────────
  Total beliefs:    5
  High confidence:  0
  Low confidence:   0
  Corrected:        0

  ── Vocabulary ──────────────────────────────────────────
  Total words:      38
  Fully acquired:   0
  Familiar:         0
  Still learning:   38
  Avg confidence:   0.22
═════════════════════════════════════════════════════════════

  Session saved. Goodbye!
```