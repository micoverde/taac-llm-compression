# Experiment 3D: Alternative Compression Methods for Code Generation

**Principal Investigator**: Dr. James Liu
**Institution**: MIT Computer Science and Artificial Intelligence Laboratory
**Date**: 2026-01-18
**Status**: Experimental Design Document (NeurIPS Submission Track)
**GitHub Issue**: #1138
**Related Experiments**: 3A (Signature Preservation), 3B (Task Classification), 3C (Perplexity Engineering)

---

## Abstract

This document presents a rigorous experimental design for comparing prompt compression algorithms in the context of code generation tasks. Our primary contribution is a novel **syntax-aware compression method (CodeCompress)** that preserves Abstract Syntax Tree (AST) structure while achieving aggressive compression ratios. We design a factorial experiment comparing five compression methods across eight compression ratios on four code generation benchmarks, with the goal of identifying whether the quality threshold r* generalizes across methods or is method-specific. This work addresses a critical gap: existing compression methods (LLMLingua-2, Selective Context) were designed for natural language, not code, leading to systematic failure modes when applied to programming tasks.

---

## 1. Research Questions

### Primary Research Questions

**RQ1: Does the quality threshold r* generalize across compression methods?**

*Hypothesis*: Different compression methods will exhibit different quality cliff thresholds (r*) due to their distinct token selection mechanisms. Methods trained on natural language (LLMLingua-2, Selective Context) will have higher r* values than code-aware methods.

*Formalization*: Let Q_m(r) denote the quality function (pass@1) for method m at compression ratio r. We define r*_m as the inflection point where:

```
r*_m = argmax_r |d²Q_m(r)/dr²|
```

We test H_0: r*_{LLMLingua-2} = r*_{CodeCompress} vs H_1: r*_{LLMLingua-2} > r*_{CodeCompress}

**RQ2: Can syntax-aware compression achieve r=0.3 with acceptable quality?**

*Hypothesis*: CodeCompress, by preserving AST-critical tokens (function definitions, type annotations, control flow), will maintain pass@1 >= 60% at r=0.3, compared to ~40% for LLMLingua-2.

**RQ3: What token categories are most predictive of compression failure?**

*Hypothesis*: Removal of function signatures and type annotations correlates most strongly with quality degradation, while removal of comments and docstrings has minimal impact.

**RQ4: Is the compression-quality tradeoff Pareto-optimal or can methods be combined?**

*Hypothesis*: Ensemble methods combining syntax-aware and perplexity-based approaches can achieve better Pareto efficiency than any single method.

### Secondary Research Questions

**RQ5**: Does the r* threshold vary by programming language (Python vs Java vs JavaScript)?

**RQ6**: How does compression time scale with input length across methods?

**RQ7**: Can we identify task categories that are inherently more compressible?

---

## 2. Algorithms to Compare

### 2.1 LLMLingua-2 (Pan et al., ACL Findings 2024)

**Mechanism**: BERT-based binary token classification trained on MeetingBank distillation data.

**Architecture**:
```
Input: Token sequence T = [t_1, ..., t_n]
Model: BERT-base-uncased (110M parameters)
Output: P(preserve | t_i, context) for each token

Compression:
1. Encode tokens with BERT
2. Classify each token as preserve/remove
3. Threshold classification scores to achieve target ratio
4. Return preserved tokens in original order
```

**Training Data**: MeetingBank dataset (meeting transcripts) with ground truth from GPT-4 distillation

**Limitations for Code**:
- Trained on natural language, not code
- "def" classified as common English word (low importance)
- Punctuation patterns differ (`:` after function definitions is critical in Python)

**Configuration Variants**:
| Variant | force_tokens | Expected r* |
|---------|-------------|-------------|
| Default | ["\n", ".", ":", "?", "!"] | 0.5-0.6 |
| Minimal | [] | 0.3-0.4 |
| Code-aware | ["def", "return", "class", ":"] | 0.4-0.5 |

**Reference**: Pan, Z., et al. (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. ACL Findings.

### 2.2 LLMLingua-1 (Jiang et al., EMNLP 2023)

**Mechanism**: Perplexity-based heuristic using causal language model (Llama-2-7B or GPT-2).

**Architecture**:
```
Input: Token sequence T = [t_1, ..., t_n]
Model: Llama-2-7B (7B parameters) or GPT-2 (124M parameters)

Compression Algorithm:
1. Compute perplexity: PPL(t_i) = exp(-log P(t_i | t_1, ..., t_{i-1}))
2. Sort tokens by perplexity (descending)
3. Keep top k tokens where k = n * ratio
4. Return tokens in original order

Intuition: High-perplexity tokens are "surprising" and informative
          Low-perplexity tokens are predictable and redundant
```

**Coarse-Grained vs Fine-Grained**:
- **Coarse**: Sentence-level selection (faster, less precise)
- **Fine**: Token-level selection (slower, more precise)

**Code Implications**:
- Keywords like "for", "if", "return" have low perplexity (common in training data)
- Variable names have high perplexity (unique to each program)
- May preserve variable names but remove keywords (problematic!)

**Reference**: Jiang, H., et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. EMNLP.

### 2.3 Selective Context (Li et al., arXiv 2023)

**Mechanism**: Self-information based selection using GPT-2.

**Architecture**:
```
Input: Token sequence T = [t_1, ..., t_n]
Model: GPT-2 (124M parameters)

Self-Information Computation:
I(t_i) = -log P(t_i | t_1, ..., t_{i-1})

Compression Algorithm:
1. Compute self-information for each token
2. Select tokens with highest self-information
3. Maintain minimum context window around selected tokens
4. Return tokens preserving local coherence

Difference from LLMLingua-1:
- Uses self-information (negative log prob) not perplexity
- Maintains local context windows
- More conservative selection strategy
```

**Theoretical Basis**: Rate-distortion theory from information theory. High self-information tokens carry more information and should be preserved.

**Code Implications**:
- Similar to LLMLingua-1: may remove keywords
- Context windows help preserve local syntax
- Better for maintaining expression coherence

**Reference**: Li, Y., et al. (2023). Compressing Context to Enhance Inference Efficiency of Large Language Models. arXiv:2310.06201.

### 2.4 Random Baseline (Control)

**Mechanism**: Uniform random token selection.

**Algorithm**:
```python
def random_compress(tokens: List[str], ratio: float) -> List[str]:
    """Random baseline compression."""
    n_keep = int(len(tokens) * ratio)
    indices = sorted(random.sample(range(len(tokens)), n_keep))
    return [tokens[i] for i in indices]
```

**Purpose**:
- Establishes lower bound for quality at each ratio
- Any method should significantly outperform random
- Helps calibrate effect sizes

**Expected Behavior**:
- Quality degrades approximately linearly with compression
- No awareness of token importance
- Should perform worst at all ratios

### 2.5 CodeCompress (Proposed Syntax-Aware Method)

**Motivation**: Existing methods fail on code because they don't understand code structure. CodeCompress leverages Abstract Syntax Tree (AST) parsing to identify and preserve syntactically critical tokens.

**Core Innovations**:
1. **AST-Guided Importance**: Use tree-sitter to parse code and assign importance based on AST node types
2. **Critical Token Protection**: Hard constraints preventing removal of structurally essential tokens
3. **Differential Compression**: Apply different strategies to code vs natural language segments
4. **Reconstruction Guarantees**: Ensure compressed output parses without syntax errors

---

## 3. CodeCompress Design Specification

### 3.1 Architecture Overview

```
                    ┌──────────────────────────────────────┐
                    │           INPUT PROMPT               │
                    │  (code + natural language context)   │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │        SEGMENT CLASSIFIER            │
                    │   Identify: code blocks, NL, mixed   │
                    └─────────────────┬────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
┌──────────▼──────────┐   ┌──────────▼──────────┐   ┌──────────▼──────────┐
│    CODE SEGMENT     │   │   MIXED SEGMENT     │   │     NL SEGMENT      │
│                     │   │                     │   │                     │
│  ┌───────────────┐  │   │  ┌───────────────┐  │   │  ┌───────────────┐  │
│  │  AST Parser   │  │   │  │  Hybrid Mode  │  │   │  │  LLMLingua-2  │  │
│  │  (tree-sitter)│  │   │  │  (AST + BERT) │  │   │  │  (aggressive) │  │
│  └───────┬───────┘  │   │  └───────┬───────┘  │   │  └───────┬───────┘  │
│          │          │   │          │          │   │          │          │
│  ┌───────▼───────┐  │   │  ┌───────▼───────┐  │   │  ┌───────▼───────┐  │
│  │ Node Tagger   │  │   │  │ Token Scorer  │  │   │  │ Token Scorer  │  │
│  │ (role assign) │  │   │  │ (combined)    │  │   │  │ (BERT-based)  │  │
│  └───────┬───────┘  │   │  └───────┬───────┘  │   │  └───────┬───────┘  │
│          │          │   │          │          │   │          │          │
│  ┌───────▼───────┐  │   │  ┌───────▼───────┐  │   │  ┌───────▼───────┐  │
│  │ Constrained   │  │   │  │ Weighted      │  │   │  │ Standard      │  │
│  │ Pruning       │  │   │  │ Pruning       │  │   │  │ Pruning       │  │
│  └───────┬───────┘  │   │  └───────┬───────┘  │   │  └───────┬───────┘  │
│          │          │   │          │          │   │          │          │
└──────────┼──────────┘   └──────────┼──────────┘   └──────────┼──────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │          RECONSTRUCTION              │
                    │    Merge segments, validate parse    │
                    └─────────────────┬────────────────────┘
                                      │
                    ┌─────────────────▼────────────────────┐
                    │          OUTPUT PROMPT               │
                    │     (compressed, syntactically       │
                    │      valid where possible)           │
                    └──────────────────────────────────────┘
```

### 3.2 AST Parsing Approach

**Parser**: tree-sitter (multi-language support, incremental parsing, fast)

**Supported Languages** (initial release):
- Python (primary)
- JavaScript/TypeScript
- Java
- Go
- Rust

**AST Node Classification**:

```python
# Node type to importance class mapping
NODE_IMPORTANCE = {
    # CRITICAL (never remove) - Importance = 1.0
    "function_definition": 1.0,
    "class_definition": 1.0,
    "method_definition": 1.0,
    "parameters": 1.0,
    "return_statement": 1.0,
    "type_annotation": 1.0,
    "decorator": 0.95,

    # HIGH (preserve unless extreme compression) - Importance = 0.8-0.9
    "if_statement": 0.9,
    "for_statement": 0.9,
    "while_statement": 0.9,
    "try_statement": 0.85,
    "assignment": 0.85,
    "import_statement": 0.85,
    "call": 0.8,

    # MEDIUM (compress at moderate ratios) - Importance = 0.5-0.7
    "expression_statement": 0.7,
    "binary_expression": 0.7,
    "identifier": 0.65,
    "string": 0.6,
    "number": 0.55,

    # LOW (aggressively compress) - Importance = 0.1-0.4
    "comment": 0.2,
    "docstring": 0.25,
    "pass_statement": 0.3,
    "ellipsis": 0.1,
}
```

**Depth-Weighted Importance**:

```python
def compute_depth_importance(node, max_depth: int = 10) -> float:
    """
    Deeper nodes (in expressions) are more specific and important.
    Shallow nodes (module level) are structural.
    """
    depth = get_node_depth(node)
    base_importance = NODE_IMPORTANCE.get(node.type, 0.5)

    # Depth scaling: deeper = more specific = more important for semantics
    # But cap at max_depth to avoid over-weighting deeply nested code
    depth_factor = min(depth / max_depth, 1.0) * 0.3 + 0.7

    return base_importance * depth_factor
```

### 3.3 Critical Node Identification

**Definition**: Critical nodes are AST elements whose removal causes:
1. Syntax errors (parse failure)
2. Semantic ambiguity (multiple valid interpretations)
3. Loss of function signature information

**Critical Token Categories**:

```python
CRITICAL_TOKENS = {
    # 1. Function/Class Definitions (MUST PRESERVE)
    "def_keywords": ["def", "class", "async", "fn", "func", "function"],

    # 2. Parameter Lists (MUST PRESERVE - signature information)
    "param_delimiters": ["(", ")", ","],
    "param_annotations": [":", "->"],

    # 3. Control Flow (MUST PRESERVE - program logic)
    "control_flow": ["if", "else", "elif", "for", "while", "return",
                     "yield", "break", "continue", "try", "except",
                     "finally", "with", "match", "case"],

    # 4. Import/Module (PRESERVE for context)
    "imports": ["import", "from", "as"],

    # 5. Operators (PRESERVE - semantic meaning)
    "comparison": ["==", "!=", "<", ">", "<=", ">=", "is", "in", "not"],
    "assignment": ["=", "+=", "-=", "*=", "/=", ":="],
    "arithmetic": ["+", "-", "*", "/", "//", "%", "**"],
    "logical": ["and", "or", "not"],

    # 6. Structural (PRESERVE - syntax validity)
    "blocks": [":", "{", "}", "[", "]"],
    "statement_end": [";", "\n"],  # Language-dependent
}

NON_CRITICAL_TOKENS = {
    # Can be aggressively compressed
    "comments": ["#", "//", "/*", "*/", "'''", '"""'],
    "docstrings": True,  # Identified by AST node type
    "whitespace": [" ", "\t"],  # Beyond minimum required
    "redundant_parens": True,  # Extra parentheses
}
```

**Critical Token Detection Algorithm**:

```python
def is_critical_token(token: str, ast_node: Node, context: Context) -> bool:
    """
    Determine if a token is critical for code correctness.

    Returns:
        True if token MUST be preserved
        False if token CAN be removed
    """
    # Rule 1: Explicit critical token lists
    for category, tokens in CRITICAL_TOKENS.items():
        if token in tokens:
            return True

    # Rule 2: Function signature tokens
    if is_in_function_signature(ast_node, context):
        return True

    # Rule 3: Type annotation tokens
    if is_type_annotation(ast_node):
        return True

    # Rule 4: First occurrence of variable names
    if is_first_variable_occurrence(token, context):
        return True

    # Rule 5: Tokens that would create syntax error if removed
    if removal_causes_syntax_error(token, ast_node, context):
        return True

    return False
```

### 3.4 Non-Critical Compression

**Principle**: Apply aggressive compression (using LLMLingua-2) to non-critical segments.

**Comment Compression**:

```python
def compress_comment(comment: str, target_ratio: float = 0.3) -> str:
    """
    Compress comment text using LLMLingua-2.
    Comments are natural language, so LLMLingua-2 works well.
    """
    # Strip comment markers
    text = strip_comment_markers(comment)

    # Apply LLMLingua-2 with aggressive settings
    compressed = llmlingua2_compress(
        text,
        rate=target_ratio,
        force_tokens=[],  # No forced preservation
        drop_consecutive=True
    )

    # Re-add comment marker
    return restore_comment_markers(compressed, comment)
```

**Docstring Compression**:

```python
def compress_docstring(docstring: str, target_ratio: float = 0.4) -> str:
    """
    Compress docstring while preserving key information.

    Priority preservation:
    1. First sentence (summary)
    2. Parameter names and types
    3. Return type description
    4. Examples (if space allows)
    """
    sections = parse_docstring(docstring)

    # Always keep: summary line
    compressed = sections.get("summary", "")

    # Conditionally keep: params (compressed)
    if target_ratio > 0.3 and "params" in sections:
        param_text = " ".join(f"{p.name}:{p.type}"
                              for p in sections["params"])
        compressed += f" Params: {param_text}"

    # Conditionally keep: return (compressed)
    if target_ratio > 0.4 and "returns" in sections:
        compressed += f" Returns: {sections['returns'].type}"

    return format_as_docstring(compressed)
```

**Identifier Compression** (optional, experimental):

```python
def compress_identifiers(code: str, target_ratio: float) -> str:
    """
    Shorten variable names while preserving uniqueness.

    WARNING: This is experimental and may hurt readability.
    Only apply at extreme ratios (< 0.3).
    """
    if target_ratio >= 0.3:
        return code  # Don't compress identifiers at moderate ratios

    # Build identifier map
    identifiers = extract_identifiers(code)

    # Generate short names (a, b, c, ..., aa, ab, ...)
    short_names = generate_short_names(len(identifiers))

    # Create mapping preserving first letter for readability
    mapping = {}
    for i, (name, freq) in enumerate(
        sorted(identifiers.items(), key=lambda x: -x[1])
    ):
        # Keep frequently used names short
        if len(name) > 3:
            mapping[name] = f"{name[0]}{i}"  # e.g., "fibonacci" -> "f0"

    # Replace in code (careful with scoping!)
    return replace_identifiers(code, mapping)
```

### 3.5 Reconstruction Algorithm

**Goal**: Produce syntactically valid output that can be parsed.

**Algorithm**:

```python
def reconstruct_compressed_code(
    original: str,
    preserved_tokens: List[Token],
    language: str = "python"
) -> str:
    """
    Reconstruct compressed code from preserved tokens.

    Guarantees:
    1. Output is syntactically valid (or best-effort valid)
    2. Token order is preserved
    3. Minimal whitespace is inserted for readability
    """
    # Step 1: Order tokens by original position
    tokens = sorted(preserved_tokens, key=lambda t: t.position)

    # Step 2: Build output with smart spacing
    output = []
    prev_token = None

    for token in tokens:
        # Determine spacing needed
        if prev_token is not None:
            space = compute_required_spacing(prev_token, token, language)
            output.append(space)

        output.append(token.text)
        prev_token = token

    result = "".join(output)

    # Step 3: Validate syntax
    if not is_syntactically_valid(result, language):
        # Attempt repair
        result = attempt_syntax_repair(result, language)

    return result


def compute_required_spacing(prev: Token, curr: Token, lang: str) -> str:
    """
    Determine minimum spacing between tokens.
    """
    # Newline rules
    if curr.text in ["def", "class", "if", "for", "while", "try"]:
        return "\n"

    # No space rules
    if prev.text in ["(", "[", "{", ".", "@"]:
        return ""
    if curr.text in [")", "]", "}", ",", ":", "."]:
        return ""

    # Space rules
    if prev.text in ["=", "==", "!=", "<", ">", "+", "-", "*", "/"]:
        return " "
    if curr.text in ["=", "==", "!=", "<", ">", "+", "-", "*", "/"]:
        return " "

    # Default: single space
    return " "


def attempt_syntax_repair(code: str, language: str) -> str:
    """
    Attempt to repair syntax errors in compressed code.

    Strategies:
    1. Add missing colons after def/if/for/while
    2. Balance parentheses/brackets
    3. Add pass to empty blocks
    """
    repairs = []

    # Missing colons
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if any(line.strip().startswith(kw) for kw in ["def", "if", "for", "while", "class"]):
            if not line.rstrip().endswith(":"):
                lines[i] = line.rstrip() + ":"
                repairs.append(f"Added colon to line {i}")

    code = "\n".join(lines)

    # Balance delimiters
    code = balance_delimiters(code)

    return code
```

### 3.6 Complete CodeCompress Pseudocode

```python
class CodeCompress:
    """
    Syntax-aware prompt compression for code generation tasks.

    Key innovations:
    1. AST-guided token importance
    2. Critical token protection
    3. Differential compression (code vs NL)
    4. Syntax-preserving reconstruction
    """

    def __init__(
        self,
        model: str = "microsoft/codebert-base",
        languages: List[str] = ["python", "javascript", "java"],
        use_codebert: bool = True,
    ):
        # Load CodeBERT for contextual importance (optional)
        if use_codebert:
            self.codebert = AutoModel.from_pretrained(model)
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            self.codebert = None
            self.tokenizer = None

        # Initialize tree-sitter parsers
        self.parsers = {
            lang: TreeSitterParser(lang) for lang in languages
        }

        # LLMLingua-2 for natural language segments
        self.nl_compressor = LLMLingua2Compressor()

        # Importance weights
        self.weights = {
            "codebert": 0.3,      # Contextual importance from CodeBERT
            "ast_depth": 0.2,     # Depth in AST (deeper = more specific)
            "ast_type": 0.3,      # Node type importance
            "role": 0.2,          # Token role (keyword, identifier, etc.)
        }

    def compress(
        self,
        prompt: str,
        ratio: float,
        language: str = "python",
        preserve_signatures: bool = True,
        compress_comments: bool = True,
    ) -> CompressedResult:
        """
        Compress a code prompt while preserving syntactic validity.

        Args:
            prompt: Input prompt (may contain code and natural language)
            ratio: Target compression ratio (0.0-1.0)
            language: Programming language for AST parsing
            preserve_signatures: Whether to always preserve function signatures
            compress_comments: Whether to compress comments/docstrings

        Returns:
            CompressedResult with compressed text and metadata
        """
        # Step 1: Segment prompt into code and natural language
        segments = self._segment_prompt(prompt, language)

        # Step 2: Process each segment
        compressed_segments = []
        for segment in segments:
            if segment.type == "code":
                compressed = self._compress_code_segment(
                    segment.text, ratio, language,
                    preserve_signatures, compress_comments
                )
            elif segment.type == "natural_language":
                compressed = self._compress_nl_segment(segment.text, ratio)
            else:  # mixed
                compressed = self._compress_mixed_segment(
                    segment.text, ratio, language
                )
            compressed_segments.append(compressed)

        # Step 3: Reconstruct
        result_text = self._reconstruct(compressed_segments, prompt)

        # Step 4: Validate and compute metrics
        actual_ratio = len(result_text) / len(prompt)
        is_valid = self._validate_syntax(result_text, language)

        return CompressedResult(
            text=result_text,
            original_length=len(prompt),
            compressed_length=len(result_text),
            target_ratio=ratio,
            actual_ratio=actual_ratio,
            is_syntactically_valid=is_valid,
            preserved_signatures=self._extract_preserved_signatures(result_text),
        )

    def _segment_prompt(
        self,
        prompt: str,
        language: str
    ) -> List[Segment]:
        """
        Segment prompt into code blocks and natural language.

        Uses heuristics:
        1. Markdown code fences (```python ... ```)
        2. Indentation patterns
        3. Keyword detection
        """
        segments = []

        # Check for markdown code fences
        fence_pattern = r'```(\w*)\n(.*?)```'
        matches = list(re.finditer(fence_pattern, prompt, re.DOTALL))

        if matches:
            # Process fenced code blocks
            prev_end = 0
            for match in matches:
                # Natural language before fence
                if match.start() > prev_end:
                    nl_text = prompt[prev_end:match.start()]
                    if nl_text.strip():
                        segments.append(Segment("natural_language", nl_text))

                # Code block
                code_text = match.group(2)
                segments.append(Segment("code", code_text))
                prev_end = match.end()

            # Remaining text
            if prev_end < len(prompt):
                remaining = prompt[prev_end:]
                if remaining.strip():
                    segments.append(Segment("natural_language", remaining))
        else:
            # No fences - use heuristics
            segments = self._segment_by_heuristics(prompt, language)

        return segments

    def _compress_code_segment(
        self,
        code: str,
        ratio: float,
        language: str,
        preserve_signatures: bool,
        compress_comments: bool,
    ) -> str:
        """
        Compress a code segment using AST-aware compression.
        """
        # Parse AST
        parser = self.parsers[language]
        try:
            tree = parser.parse(code)
        except ParseError:
            # Fallback to heuristic compression if parse fails
            return self._fallback_compress(code, ratio)

        # Extract and classify tokens
        tokens = self._extract_tokens_with_ast(code, tree)

        # Compute importance scores
        for token in tokens:
            token.importance = self._compute_importance(token)
            token.is_critical = self._is_critical(token, preserve_signatures)

        # Handle comments/docstrings
        if compress_comments:
            tokens = self._compress_comments_in_tokens(tokens, ratio)

        # Select tokens to preserve
        preserved = self._select_tokens(tokens, ratio)

        # Reconstruct
        return self._reconstruct_code(preserved, language)

    def _compute_importance(self, token: EnhancedToken) -> float:
        """
        Compute token importance as weighted sum of factors.

        importance = w1 * codebert_score
                   + w2 * ast_depth_score
                   + w3 * ast_type_score
                   + w4 * role_score
        """
        scores = {}

        # CodeBERT contextual importance
        if self.codebert is not None:
            scores["codebert"] = self._codebert_importance(token)
        else:
            scores["codebert"] = 0.5  # Default

        # AST depth (normalized)
        scores["ast_depth"] = min(token.ast_depth / 10.0, 1.0)

        # AST node type importance
        scores["ast_type"] = NODE_IMPORTANCE.get(token.ast_node_type, 0.5)

        # Token role importance
        role_importance = {
            "keyword": 0.9,
            "identifier": 0.7,
            "operator": 0.8,
            "literal": 0.6,
            "punctuation": 0.75,
            "comment": 0.2,
            "whitespace": 0.1,
        }
        scores["role"] = role_importance.get(token.role, 0.5)

        # Weighted sum
        importance = sum(
            self.weights[key] * scores[key]
            for key in self.weights
        )

        return importance

    def _select_tokens(
        self,
        tokens: List[EnhancedToken],
        ratio: float
    ) -> List[EnhancedToken]:
        """
        Select tokens to preserve based on importance and constraints.

        Algorithm:
        1. Always include critical tokens
        2. Sort remaining by importance
        3. Fill to target ratio
        """
        # Separate critical and non-critical
        critical = [t for t in tokens if t.is_critical]
        non_critical = [t for t in tokens if not t.is_critical]

        # Sort non-critical by importance
        non_critical_sorted = sorted(
            non_critical,
            key=lambda t: t.importance,
            reverse=True
        )

        # Calculate target token count
        total_tokens = len(tokens)
        target_count = int(total_tokens * ratio)

        # Start with critical tokens
        preserved = list(critical)

        # Fill with highest importance non-critical tokens
        remaining_slots = max(0, target_count - len(preserved))
        preserved.extend(non_critical_sorted[:remaining_slots])

        # Sort by original position for reconstruction
        preserved.sort(key=lambda t: t.position)

        return preserved

    def _compress_nl_segment(self, text: str, ratio: float) -> str:
        """
        Compress natural language using LLMLingua-2.
        """
        return self.nl_compressor.compress(
            text,
            rate=ratio,
            force_tokens=[],
            drop_consecutive=True
        )

    def _validate_syntax(self, code: str, language: str) -> bool:
        """
        Check if compressed code is syntactically valid.
        """
        try:
            parser = self.parsers[language]
            tree = parser.parse(code)
            # Check for ERROR nodes in tree
            return not self._has_error_nodes(tree)
        except:
            return False
```

---

## 4. Experimental Design

### 4.1 Factorial Design

**Independent Variables**:

| Factor | Levels | Description |
|--------|--------|-------------|
| **Algorithm** | 5 | LLMLingua-2, LLMLingua-1, Selective Context, Random, CodeCompress |
| **Compression Ratio** | 8 | 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 |
| **Benchmark** | 4 | HumanEval, MBPP, DS-1000, CodeContests |
| **Language** | 3 | Python, JavaScript, Java (subset analysis) |

**Full Factorial**: 5 × 8 × 4 = 160 experimental conditions (primary)
**Extended**: 5 × 8 × 4 × 3 = 480 conditions (with language variation)

**Dependent Variables**:
- Primary: pass@1, pass@3
- Secondary: syntactic validity rate, compression time, actual achieved ratio

### 4.2 Sample Size and Power Analysis

**Target Effect Size**: Detect 10% absolute difference in pass@1 (e.g., 40% vs 50%)

**Power Analysis** (using G*Power):
```
Test: Two-proportion z-test
Alpha: 0.05 (Bonferroni-corrected for 10 primary comparisons: 0.005)
Power: 0.80
Effect size (h): 0.20 (small-medium)
Required n per group: ~400 samples
```

**Practical Sample Size**:
- HumanEval: 164 problems × 5 trials = 820 samples
- MBPP: 500 problems (subset) × 3 trials = 1,500 samples
- DS-1000: 500 problems (subset) × 3 trials = 1,500 samples
- CodeContests: 165 problems × 3 trials = 495 samples

**Total**: ~4,300 samples per algorithm-ratio combination

### 4.3 Evaluation Protocol

```python
def run_experiment(
    algorithm: CompressionMethod,
    benchmark: Benchmark,
    ratio: float,
    model: str = "gpt-4o-mini",  # Cost-effective
    n_samples: int = 3,
    seed: int = 42,
) -> ExperimentResult:
    """
    Run compression experiment for one algorithm-benchmark-ratio combination.
    """
    results = []

    for problem in tqdm(benchmark.problems):
        # Compress prompt
        start_time = time.time()
        compressed = algorithm.compress(problem.prompt, ratio=ratio)
        compression_time = time.time() - start_time

        # Compute actual ratio
        actual_ratio = len(compressed.text) / len(problem.prompt)

        # Generate solutions
        solutions = generate_solutions(
            model=model,
            prompt=compressed.text,
            n=n_samples,
            temperature=0.2,
            seed=seed,
        )

        # Execute tests
        test_results = [
            execute_tests(solution, problem.test_cases)
            for solution in solutions
        ]

        # Record result
        results.append({
            "problem_id": problem.id,
            "algorithm": algorithm.name,
            "target_ratio": ratio,
            "actual_ratio": actual_ratio,
            "compression_time_ms": compression_time * 1000,
            "solutions": solutions,
            "test_results": test_results,
            "pass_at_1": test_results[0] if test_results else False,
            "pass_at_3": any(test_results[:3]),
            "syntactic_validity": sum(is_valid_python(s) for s in solutions) / len(solutions),
            "original_tokens": count_tokens(problem.prompt),
            "compressed_tokens": count_tokens(compressed.text),
            "preserved_signatures": compressed.preserved_signatures,
        })

    return ExperimentResult(
        algorithm=algorithm.name,
        benchmark=benchmark.name,
        ratio=ratio,
        results=results,
        aggregate={
            "pass_at_1": np.mean([r["pass_at_1"] for r in results]),
            "pass_at_3": np.mean([r["pass_at_3"] for r in results]),
            "avg_compression_time_ms": np.mean([r["compression_time_ms"] for r in results]),
            "avg_actual_ratio": np.mean([r["actual_ratio"] for r in results]),
            "syntactic_validity": np.mean([r["syntactic_validity"] for r in results]),
        }
    )
```

### 4.4 Blocking and Randomization

**Blocking Factors**:
- Problem difficulty (easy/medium/hard based on baseline pass@1)
- Problem length (short/medium/long based on token count)
- Problem type (string manipulation, algorithm, data structure, etc.)

**Randomization**:
- Problem order randomized within each algorithm-ratio condition
- Solution generation uses fixed seeds for reproducibility
- API calls batched to minimize variance from model state

### 4.5 Pilot Study Protocol

**Phase 1 Pilot** (before full experiment):
1. Run each algorithm on HumanEval subset (n=50) at r=0.3, 0.5, 0.7
2. Validate implementation correctness
3. Estimate variance for power analysis refinement
4. Identify any algorithmic bugs or edge cases

**Pilot Success Criteria**:
- All algorithms complete without errors
- Variance in pass@1 < 15% across replicates
- Compression times < 5 seconds per problem
- At least one algorithm achieves pass@1 > 50% at r=0.3

---

## 5. Metrics

### 5.1 Primary Metrics

**pass@k**: Probability of at least one correct solution in k attempts

```python
def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric (Chen et al., 2021).

    Args:
        n: Total number of samples generated
        c: Number of correct samples
        k: k for pass@k

    Returns:
        pass@k probability
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
```

**Quality Threshold r*** (novel metric):

```python
def compute_quality_threshold(
    ratios: List[float],
    pass_at_1: List[float],
    baseline_pass_at_1: float,
    threshold_fraction: float = 0.9,
) -> float:
    """
    Compute r*, the compression ratio where quality drops below threshold.

    Definition: r* = max{r : Q(r) >= threshold_fraction * Q(1.0)}

    Uses piecewise linear interpolation to find exact threshold.
    """
    target_quality = threshold_fraction * baseline_pass_at_1

    # Sort by ratio (descending)
    sorted_data = sorted(zip(ratios, pass_at_1), reverse=True)

    # Find transition point
    for i in range(len(sorted_data) - 1):
        r1, q1 = sorted_data[i]
        r2, q2 = sorted_data[i + 1]

        if q1 >= target_quality and q2 < target_quality:
            # Linear interpolation
            slope = (q1 - q2) / (r1 - r2)
            r_star = r2 + (target_quality - q2) / slope
            return r_star

    # If quality never drops below threshold
    return min(ratios)
```

### 5.2 Secondary Metrics

**Compression Speed**:
```python
def measure_compression_speed(
    algorithm: CompressionMethod,
    prompts: List[str],
    n_trials: int = 3,
) -> SpeedMetrics:
    """Measure compression throughput."""
    times = []
    tokens = []

    for prompt in prompts:
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = algorithm.compress(prompt, ratio=0.5)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            tokens.append(count_tokens(prompt))

    return SpeedMetrics(
        mean_time_ms=np.mean(times) * 1000,
        std_time_ms=np.std(times) * 1000,
        tokens_per_second=np.sum(tokens) / np.sum(times),
        p95_time_ms=np.percentile(times, 95) * 1000,
    )
```

**Syntactic Validity Rate**:
```python
def syntactic_validity_rate(solutions: List[str], language: str) -> float:
    """Fraction of solutions that parse without errors."""
    valid = 0
    for solution in solutions:
        try:
            if language == "python":
                ast.parse(solution)
            elif language == "javascript":
                esprima.parseScript(solution)
            valid += 1
        except:
            pass
    return valid / len(solutions)
```

**Token Preservation Analysis**:
```python
def analyze_token_preservation(
    original: str,
    compressed: str,
) -> TokenPreservationMetrics:
    """Analyze which token categories were preserved/removed."""
    original_tokens = tokenize_with_roles(original)
    compressed_tokens = set(tokenize(compressed))

    by_role = defaultdict(lambda: {"preserved": 0, "removed": 0})

    for token in original_tokens:
        role = token.role
        if token.text in compressed_tokens:
            by_role[role]["preserved"] += 1
        else:
            by_role[role]["removed"] += 1

    preservation_rates = {
        role: counts["preserved"] / (counts["preserved"] + counts["removed"])
        for role, counts in by_role.items()
    }

    return TokenPreservationMetrics(
        preservation_by_role=preservation_rates,
        signature_preserved=check_signature_preserved(original, compressed),
        keywords_preserved=preservation_rates.get("keyword", 0),
        identifiers_preserved=preservation_rates.get("identifier", 0),
    )
```

### 5.3 Metric Collection Matrix

| Metric | Per-Problem | Per-Ratio | Per-Algorithm | Aggregate |
|--------|-------------|-----------|---------------|-----------|
| pass@1 | Yes | Yes | Yes | Yes |
| pass@3 | Yes | Yes | Yes | Yes |
| r* | No | No | Yes | Yes |
| Compression time | Yes | Yes | Yes | Yes |
| Actual ratio | Yes | Yes | Yes | Yes |
| Syntactic validity | Yes | Yes | Yes | Yes |
| Token preservation | Yes (subset) | Yes | Yes | Yes |
| Signature preserved | Yes | Yes | Yes | Yes |

---

## 6. Statistical Analysis

### 6.1 Algorithm Comparison

**Primary Test**: Friedman test (non-parametric) for comparing 5 algorithms across all conditions.

```python
def algorithm_comparison_test(results: Dict[str, List[float]]) -> StatResult:
    """
    Compare algorithms using Friedman test with Nemenyi post-hoc.

    Args:
        results: Dict mapping algorithm name to list of pass@1 values

    Returns:
        Statistical test results
    """
    from scipy.stats import friedmanchisquare
    from scikit_posthocs import posthoc_nemenyi_friedman

    # Prepare data matrix
    algorithms = list(results.keys())
    data = np.array([results[alg] for alg in algorithms]).T

    # Friedman test
    stat, p_value = friedmanchisquare(*data.T)

    # Post-hoc if significant
    if p_value < 0.05:
        posthoc = posthoc_nemenyi_friedman(data)
    else:
        posthoc = None

    return StatResult(
        test="Friedman",
        statistic=stat,
        p_value=p_value,
        significant=p_value < 0.05,
        posthoc=posthoc,
    )
```

**Secondary Test**: Paired t-tests for specific comparisons (Bonferroni-corrected).

| Comparison | Hypothesis | Expected Direction |
|------------|------------|-------------------|
| CodeCompress vs LLMLingua-2 | CodeCompress > LLMLingua-2 at r=0.3 | + |
| LLMLingua-1 vs LLMLingua-2 | Different r* thresholds | ? |
| Selective Context vs LLMLingua-1 | Similar performance | = |
| All methods vs Random | All > Random | + |

### 6.2 Threshold Detection

**Method**: Piecewise regression with breakpoint detection.

```python
def detect_quality_cliff(
    ratios: np.ndarray,
    qualities: np.ndarray,
) -> ThresholdResult:
    """
    Detect quality cliff using piecewise linear regression.

    Model: Q(r) = {a1 + b1*r  if r >= r*
                  {a2 + b2*r  if r < r*

    Fit using least squares with breakpoint optimization.
    """
    from scipy.optimize import minimize_scalar

    def fit_piecewise(breakpoint):
        mask_high = ratios >= breakpoint
        mask_low = ratios < breakpoint

        if sum(mask_high) < 2 or sum(mask_low) < 2:
            return np.inf

        # Fit two linear models
        model_high = np.polyfit(ratios[mask_high], qualities[mask_high], 1)
        model_low = np.polyfit(ratios[mask_low], qualities[mask_low], 1)

        # Compute RSS
        pred_high = np.polyval(model_high, ratios[mask_high])
        pred_low = np.polyval(model_low, ratios[mask_low])

        rss = np.sum((qualities[mask_high] - pred_high)**2)
        rss += np.sum((qualities[mask_low] - pred_low)**2)

        return rss

    # Find optimal breakpoint
    result = minimize_scalar(
        fit_piecewise,
        bounds=(min(ratios) + 0.05, max(ratios) - 0.05),
        method='bounded'
    )

    return ThresholdResult(
        r_star=result.x,
        quality_above=np.mean(qualities[ratios >= result.x]),
        quality_below=np.mean(qualities[ratios < result.x]),
        slope_change=compute_slope_change(ratios, qualities, result.x),
    )
```

### 6.3 Heterogeneity Analysis

**Research Question**: Does r* vary significantly across benchmarks and problem types?

**Method**: Mixed-effects meta-regression.

```python
def heterogeneity_analysis(
    results: List[ExperimentResult],
) -> HeterogeneityResult:
    """
    Analyze heterogeneity in r* across conditions using meta-regression.

    Model: r*_ij = mu + alpha_i + beta_j + epsilon_ij

    Where:
        i = algorithm
        j = benchmark
        alpha_i = algorithm fixed effect
        beta_j = benchmark random effect
    """
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM

    # Prepare data
    data = []
    for result in results:
        r_star = compute_quality_threshold(result)
        data.append({
            "r_star": r_star,
            "algorithm": result.algorithm,
            "benchmark": result.benchmark,
            "avg_problem_length": result.avg_problem_length,
            "problem_type": result.problem_type,
        })

    df = pd.DataFrame(data)

    # Fit mixed-effects model
    model = MixedLM.from_formula(
        "r_star ~ algorithm + avg_problem_length + C(problem_type)",
        groups="benchmark",
        data=df
    )
    result = model.fit()

    # Compute I² heterogeneity statistic
    Q = result.scale  # Between-study variance
    tau2 = result.cov_re.iloc[0, 0]  # Random effect variance
    I2 = tau2 / (tau2 + Q) * 100

    return HeterogeneityResult(
        I2=I2,
        algorithm_effects=result.params,
        benchmark_variance=tau2,
        significant_predictors=result.pvalues[result.pvalues < 0.05].index.tolist(),
    )
```

### 6.4 Multiple Comparisons Correction

**Strategy**: Bonferroni-Holm stepdown procedure.

```python
def bonferroni_holm_correction(p_values: Dict[str, float]) -> Dict[str, dict]:
    """
    Apply Bonferroni-Holm correction for multiple comparisons.
    """
    comparisons = sorted(p_values.items(), key=lambda x: x[1])
    n = len(comparisons)

    results = {}
    rejected = False

    for i, (comparison, p) in enumerate(comparisons):
        threshold = 0.05 / (n - i)

        if not rejected and p < threshold:
            significant = True
        else:
            significant = False
            rejected = True  # Stop rejecting once one fails

        results[comparison] = {
            "p_value": p,
            "adjusted_threshold": threshold,
            "significant": significant,
        }

    return results
```

---

## 7. Hardware Requirements

### 7.1 GPU Requirements by Algorithm

| Algorithm | Model | VRAM Required | Recommended GPU |
|-----------|-------|---------------|-----------------|
| **LLMLingua-2** | BERT-base | 2-4 GB | T4 / RTX 3080 |
| **LLMLingua-1** | Llama-2-7B | 14-16 GB | A100 40GB / RTX 4090 |
| **LLMLingua-1** (alt) | GPT-2 | 2-4 GB | T4 / RTX 3080 |
| **Selective Context** | GPT-2-large | 4-6 GB | T4 / RTX 3080 |
| **Random Baseline** | None | 0 | CPU only |
| **CodeCompress** | CodeBERT + tree-sitter | 4-6 GB | T4 / RTX 3080 |
| **Generation Model** | GPT-4o-mini API | N/A | Cloud API |

### 7.2 Memory Requirements

| Phase | RAM Required | Storage | Notes |
|-------|--------------|---------|-------|
| Data loading | 16 GB | 50 GB | Benchmarks + cache |
| Compression | 32 GB | 100 GB | Model weights + batches |
| Evaluation | 16 GB | 200 GB | Results + logs |
| Analysis | 64 GB | 50 GB | Statistical computing |

### 7.3 Estimated Compute Budget

| Resource | Quantity | Unit Cost | Total |
|----------|----------|-----------|-------|
| A100 40GB (LLMLingua-1) | 24 hours | $2.50/hr | $60 |
| T4 GPU (other algorithms) | 72 hours | $0.35/hr | $25 |
| GPT-4o-mini API | 5M tokens | $0.15/1M | $0.75 |
| Storage (Azure Blob) | 500 GB/month | $0.02/GB | $10 |
| **Total** | | | **~$96** |

### 7.4 Recommended Infrastructure

**Option A: Azure VM (Recommended)**
```
VM: Standard_NC24ads_A100_v4
GPU: 1x A100 40GB
vCPU: 24
RAM: 220 GB
Cost: ~$3.50/hour
Estimated runtime: 24-48 hours
Total: $84-168
```

**Option B: Google Colab Pro+ (Budget)**
```
GPU: A100 40GB (when available)
RAM: 52 GB high-RAM runtime
Cost: $50/month
Limitation: 24-hour session limits
```

**Option C: Lambda Labs (Spot)**
```
Instance: 1x A100 (40GB)
vCPU: 30
RAM: 200 GB
Cost: $1.10/hour (spot)
Estimated runtime: 30 hours
Total: $33 (if no interruptions)
```

---

## 8. References

### Primary Methods Papers

1. **Pan, Z., Wu, S., Shen, M., Huang, K., Liu, X., Wu, Z., & Xing, E.** (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. *Findings of the Association for Computational Linguistics: ACL 2024*. https://arxiv.org/abs/2403.12968

2. **Jiang, H., Wu, Q., Lin, C., Yang, Y., & Qiu, L.** (2023). LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. *Proceedings of EMNLP 2023*. https://arxiv.org/abs/2310.05736

3. **Li, Y., Bubeck, S., Eldan, R., Del Giorno, A., Gunasekar, S., & Lee, Y. T.** (2023). Textbooks Are All You Need II: phi-1.5 technical report. *arXiv preprint arXiv:2309.05463*.

4. **Li, Y., Chen, H., Nan, G., Wu, Y., Wang, Y., & Xing, E.** (2023). Compressing Context to Enhance Inference Efficiency of Large Language Models. *arXiv preprint arXiv:2310.06201*.

### Code Models and Benchmarks

5. **Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H.P., et al.** (2021). Evaluating Large Language Models Trained on Code. *arXiv preprint arXiv:2107.03374*. (HumanEval benchmark)

6. **Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., et al.** (2021). Program Synthesis with Large Language Models. *arXiv preprint arXiv:2108.07732*. (MBPP benchmark)

7. **Feng, Z., Guo, D., Tang, D., Duan, N., Feng, X., Gong, M., Shou, L., Qin, B., Liu, T., Jiang, D., & Zhou, M.** (2020). CodeBERT: A Pre-Trained Model for Programming and Natural Languages. *Findings of EMNLP 2020*. https://arxiv.org/abs/2002.08155

8. **Lai, Y., Li, C., Wang, Y., Zhang, T., et al.** (2022). DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation. *arXiv preprint arXiv:2211.11501*.

### Statistical Methods

9. **Demsar, J.** (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. *Journal of Machine Learning Research*, 7, 1-30.

10. **Higgins, J.P., Thompson, S.G., Deeks, J.J., & Altman, D.G.** (2003). Measuring inconsistency in meta-analyses. *BMJ*, 327(7414), 557-560. (I² statistic)

### Parsing and AST

11. **Tree-sitter Contributors.** (2023). Tree-sitter: An incremental parsing system for programming tools. https://tree-sitter.github.io/tree-sitter/

12. **Allamanis, M., Barr, E.T., Devanbu, P., & Sutton, C.** (2018). A Survey of Machine Learning for Big Code and Naturalness. *ACM Computing Surveys*, 51(4), 1-37.

### Prior Work (Plexor Research)

13. **Plexor Research Team.** (2025). Compression Floor Investigation: Breaking the 8x Barrier. *Internal Technical Report*.

14. **Plexor Research Team.** (2026). Signature Preservation in Compressed Code Prompts. *Working Paper*.

---

## Appendix A: Implementation Checklist

### Pre-Experiment Setup

- [ ] Download and validate all benchmarks (HumanEval, MBPP, DS-1000, CodeContests)
- [ ] Install LLMLingua-2, set up environment
- [ ] Implement LLMLingua-1 wrapper (GPT-2 variant for cost)
- [ ] Implement Selective Context wrapper
- [ ] Implement CodeCompress with tree-sitter
- [ ] Validate API access for GPT-4o-mini
- [ ] Set up results storage (Azure Blob or local)
- [ ] Create logging and checkpointing infrastructure

### Pilot Study

- [ ] Run HumanEval subset (n=50) with all algorithms at r=0.3, 0.5, 0.7
- [ ] Verify no algorithmic errors
- [ ] Validate metric computation
- [ ] Estimate variance and refine sample sizes

### Main Experiment

- [ ] Full HumanEval evaluation (164 problems × 5 algorithms × 8 ratios)
- [ ] Full MBPP evaluation (500 problems × 5 algorithms × 8 ratios)
- [ ] DS-1000 subset (500 problems × 5 algorithms × 4 key ratios)
- [ ] CodeContests subset (165 problems × 5 algorithms × 4 key ratios)

### Analysis

- [ ] Compute pass@1, pass@3 for all conditions
- [ ] Detect r* threshold for each algorithm
- [ ] Run Friedman test and post-hoc comparisons
- [ ] Analyze token preservation patterns
- [ ] Generate Pareto frontier plots
- [ ] Write results section

---

## Appendix B: Expected Timeline

| Week | Phase | Tasks | Deliverables |
|------|-------|-------|--------------|
| 1 | Setup | Environment, data, baseline implementations | Working pipeline |
| 2 | CodeCompress | Implement and validate CodeCompress | Validated algorithm |
| 3 | Pilot | Run pilot study, validate metrics | Pilot report |
| 4 | Main Exp | HumanEval + MBPP full runs | Raw results |
| 5 | Main Exp | DS-1000 + CodeContests runs | Complete results |
| 6 | Analysis | Statistical analysis, visualizations | Figures, tables |
| 7 | Writing | Results section, discussion | Paper draft |
| 8 | Revision | Address internal review | Submission-ready |

---

## Appendix C: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| LLMLingua-1 (Llama-2-7B) too slow | Medium | Medium | Use GPT-2 variant instead |
| CodeCompress parsing fails | Low | High | Fallback to regex-based extraction |
| API rate limits | Medium | Medium | Implement exponential backoff, caching |
| High variance in pass@1 | Medium | High | Increase samples, use bootstrapping |
| No method achieves r=0.3 with quality | Medium | High | Reframe as identifying practical limits |
| GPU OOM errors | Low | Medium | Reduce batch sizes, gradient checkpointing |

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
*Author: Dr. James Liu*
*For: NeurIPS 2026 Submission*
