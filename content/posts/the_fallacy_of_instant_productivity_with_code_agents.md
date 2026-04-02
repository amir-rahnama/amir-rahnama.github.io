---
title: "Code Agents and the Myth of Instant Productivity"
date: 2026-04-02
tags: ["Code Agents", "AI-assisted Development", "Productivity"]
categories: ["Software Engineering", "AI Tools"]
draft: false
---

## Introduction

The usual take about the growth of code agents in tech teams includes two points: that the engineer’s productivity has increased by a factor of X using code agents, and that the bottleneck is now in the review and verification process.

As a result, the claim is that the bottleneck is in verification. By verification, people usually mean that code is generated but then needs to go through the human verification process for quality control. Since a machine can generate code faster than a human can read, PRs now accumulate in codebases faster than developers can review and merge them.  

In this post, I will argue that this argument is only partially true. Defining productivity based on the number of lines of code generated and accepted into the codebase overlooks the important parts of the software engineering process. The bottleneck is not producing code, but understanding it and the problem it solves.

---

## What Makes a Software Engineer Productive?

To understand this, let us discuss the tasks that comprise a productive software engineer. Broadly speaking, productivity can be about finding the right solution to different problems. However, in software engineering, no problem is standalone — problems are often stacked and interconnected. This is why a system can suffer when one of its components is suboptimal. Problems are context-specific, and this is why teams of developers usually work together.

### Key Tasks for Engineers and Verifiers

When you work as an engineer, you need to be competent in using the technology that solves your problem. You must define the problem clearly and break down the solution. The main tasks are:

1. **Find the problem**  
2. **Write code that is optimal** to solve the problem  
3. **Write verifiable code**: ensure the code is clear and understandable  
4. **Refactor and reduce complexity**  
5. **Write code that can scale** (in time and space complexity)  

LLMs and Agents are great at tasks 1, 2, and 4 — if the problem is clearly defined. They are less effective at tasks 3 and 5, which rely heavily on human understanding and broader context.

---

## How Agent-Assisted Development Works

The development process with agents is different from traditional methods. You start with a solution presented by the agent, then have two choices:

1. Continue prompting to incrementally improve the solution  
2. Start an entirely new prompt/session to generate a new solution  

Incremental improvement may lead to locally optimal solutions, while starting over can be more costly in terms of tokens. Performance also depends on the model and repository/tool quality — pricier models and structured codebases produce better results.

### Limitations of Context

Solutions are often optimized only within a local context. For example, if your codebase uses Django or FastAPI, the agent rarely suggests switching frameworks, even if that could solve underlying issues. Refactoring requires careful checks and has limitations because it involves understanding large portions of the codebase. In many cases, having a solution is less important than having the **most optimal solution**.

---

## The Verification Challenge

Even if we generate a partially optimal solution, verifying it is not easy. You cannot fully verify code unless you have understood it and the engineer’s thought process. Prompts and generation conditions are unknown to verifiers, making them more like editors adjusting a novel draft than authors of a book. Large changes are difficult without consequences.

Another issue is **model/agent sensitivity**: the same prompt can produce different outputs depending on the model, making verification inconsistent.

---

## Growing as an Engineer

One crucial aspect of software engineering is growth. Before LLMs, developers used forums like Stack Overflow to learn collaboratively. Reviewing multiple answers, helping others, and reflecting on solutions provided deep learning. Engineers began by writing imperfect code, learning from debugging, and iteratively improving solutions.

Agent-assisted development risks turning engineers into passive consumers of solutions. Developers may rely on generated code without developing the reasoning skills needed to solve new problems independently.

---

## The Analogy: Authors and Drafts

You cannot make someone a distinguished author by handing them the first draft of Harry Potter. They might edit the draft or add subplots, but the process of becoming J.K. Rowling — developing skill and intuition — cannot be bypassed. Similarly, engineers must engage deeply with problems and solutions to truly grow.

---

## Conclusion

The real bottleneck of agent-assisted development is **not code generation**, but understanding the problems and solutions produced. If the author or verifier of the code is mostly a machine, how can we ensure proper understanding? In the long run, this threatens both engineers and verifiers, and may slow growth in expertise despite apparent productivity gains.