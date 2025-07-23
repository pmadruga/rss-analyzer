---
title: "Quantization-Aware Training of jina-embeddings-v4"
date: 2025-07-02
draft: false
weight: 2
url: "/articles/quantization-aware-training-of-jina-embeddings-v4/"
tags:
  - "jina.ai"
  - "research"
  - "ai-analysis"
summary: "Quantization-Aware Training of jina-embeddings-v4"
params:
  original_url: "https://jina.ai/news/quantization-aware-training-of-jina-embeddings-v4/"
  article_id: 2
  domain: "jina.ai"
---

# Quantization-Aware Training of jina-embeddings-v4

{{< callout type="info" >}}
**Original Source:** [jina.ai](https://jina.ai/news/quantization-aware-training-of-jina-embeddings-v4/)
**Analyzed:** 2025-07-02
**AI Provider:** anthropic
{{< /callout >}}

## Fine-Tuning Improves Performance

Quantization-aware training (QAT) with fine-tuning significantly improved the performance compared to post-training quantization (PTQ).

2.

## Quantization Level Impact

Less aggressive quantization (e.g., 4-bit) generally performed better than more aggressive methods (e.g., binary). However, there was no significant difference between 8-bit and 4-bit quantization.

3.

## Scaling Methods

The rolling average scaling method outperformed the min/max approach, indicating that using scaling values relative to the data works better.

4.

## Asymmetric Quantization

Leaving query vectors unquantized improved performance in binary quantization cases.

**Technical Approach:** The technical approach involved several key components working together to achieve the quantization and evaluation of the embedding models:

1.

## Quantization Levels

The researchers experimented with different levels of quantization:
   -

## 8-bit integers

Reducing floating-point values to a range of -128 to 127.
   -

## 4-bit integers

Mapping values to a range of -8 to 7.
   -

## Trinary Quantization

Mapping values to -1, 0, or 1.
   -

## Binary Quantization

Converting values to either -1 or 1 using the torch.sign datatype.

2.

## Scaling Techniques

Two scaling techniques were used to normalize the values:
   -

## Min/Max Scaling

Identifying the maximum and minimum values in each batch.
   -

## Rolling Averaging

Calculating a moving average of the mean and standard deviation of vector components.

3.

## Fine-Tuning with Straight-Through Estimation

For Output QAT, the model was fine-tuned by reversing the quantization process to restore full precision, calculating the loss, and using that to fine-tune the model. This process involved 10,000 steps, with checkpoints saved every 500 steps.

4.

## Asymmetric Quantization

The researchers tested the impact of quantizing query vectors versus leaving them unquantized to understand the trade-offs in performance and storage.

5.

## Evaluation Metrics

The NanoBEIR benchmark was used to evaluate the performance of the quantized models. This benchmark measures the retrieval accuracy of the models by comparing the cosine similarity between vectors.

These technical components were chosen to systematically reduce the size of embedding vectors while maintaining or improving the model's performance. The combination of quantization levels, scaling techniques, and fine-tuning methods allowed the researchers to explore different trade-offs and optimizations.

**Methodology:** The research methodology involved several key steps to study the impact of quantization on embedding models, specifically focusing on making the models more efficient without losing precision. Here’s a breakdown of the process:

1.

## Baseline Establishment

The researchers started with a baseline model, jina-embeddings-v4, which produces high-precision floating-point vectors. This model was used as a reference point to compare the effects of different quantization techniques.

2.

## Quantization Techniques

Four main quantization techniques were considered:
   -

## Post-Training Quantization (PTQ)

This involves rounding off the floating-point values produced by the model to reduce their size.
   -

## Output Quantization-Aware Training (Output QAT)

This fine-tunes the model to produce optimal reduced-precision vectors, focusing only on the output.
   -

## Full Quantization-Aware Training (Full QAT)

This reduces the precision of the model weights and then fine-tunes the model for better performance.
   -

## Distillation

This involves training a new quantized model from an existing unquantized one.

3.

## Experimental Conditions

The study focused on PTQ and Output QAT. The baseline model's vectors were quantized to different levels (8-bit, 4-bit, trinary, and binary) and the performance was evaluated.

4.

## Scaling Methods

Two scaling methods were used to normalize the values for quantization:
   -

## Min/Max

Identifying the highest and lowest vector components in each batch.
   -

## Rolling Averaging

Calculating the average and standard deviation of vector components across batches.

5.

## Fine-Tuning

For Output QAT, the model was fine-tuned using straight-through estimation, which reverses the quantization process to calculate the loss and fine-tune the model.

6.

## Asymmetric Quantization

The researchers tested both quantizing the query vectors and leaving them unquantized to see the impact on performance.

7.

## Evaluation

The performance of each condition was evaluated using the NanoBEIR benchmark, which measures the retrieval accuracy of the quantized models.


---

{{< callout type="note" >}}
This analysis was generated using the Feynman technique, where the AI takes on the role of the paper's author and explains the research using simple language and analogies to make complex concepts accessible.
{{< /callout >}}
