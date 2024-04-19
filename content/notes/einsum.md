---
title: "Einsum"
date: 2024-04-18
showReadingTime: false
---

## signature
`einsum(equation, operand)` where `equation` is a string representing the Einstein summation and `operands` is a sequence of tensors.

## how its work?
- free indices: indices specified in output
- summation indices: all other indices in input not in output
- example: `np.einsum("ik, kj -> ij", A, B)`, free indices = `i`, `j` and summation index = `k`

## examples
`import torch`  
`x = torch.rand((2, 3))`
### permutation
`torch.einsum("ij -> ji", x)`
### summation
`torch.einsum("ij ->", x)`
### column sum
`torch.einsum("ij -> j", x)`
### row sum
`torch.einsum("ij -> i", x)`
### matrix-vector multiplication
`v = torch.rand((1, 3))`  
`torch.einsum("ij, kj -> ik", x, v)`
### matrix-matrix multiplication
`torch.einsum("ij, kj -> ik", x, x)`