# queries

Test questions against the default corpus (Attention Is All You Need). Grouped by what they exercise — useful for manually verifying retrieval behaviour or seeing the agent's decisions in Auto mode.

## direct lookup
```
What is multi-head attention?
What optimizer did the authors use?
How many attention heads does the base model use?
```

## jargon
BM25 matters here as rare tokens don't embed well.
```
What is the formula for scaled dot-product attention?
What is the dimensionality of the key and query vectors?
What is label smoothing and why did the authors use it?
```

## paraphrased
Wording differs from the source; rerank or HyDE helps.
```
How does the model pay attention to different positions simultaneously?
What makes this architecture better than RNNs?
Why don't the authors use recurrence?
```

## multi-hop synthesis
Spans multiple sections; multi-query rewrite helps.
```
Compare the encoder and decoder stacks — how are they similar and how do they differ?
How many parameters are in the base model vs the big model?
What regularisation techniques did the authors use and why?
```

## expected refusals
The corpus doesn't contain answers to these — the model should say so plainly.
```
When did this paper win the Turing award?
Who peer-reviewed the paper before publication?
```
