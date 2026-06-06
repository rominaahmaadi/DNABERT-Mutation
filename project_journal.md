**Day 1** — I downloaded the ClinVar database from NCBI and filtered it down to only variants with a clear binary label: Pathogenic or Benign. 

**Day 2** — ClinVar gives the positions, not sequences. 
I used BioPython to query NCBI's Entrez API and fetch 513 nucleotides of DNA around each variant. 
I learned what flanking sequences are and why the context around a mutation matters more than the mutation itself.

**Day 3** — I analyzed the dataset I built. All sequences were exactly 513 nucleotides, the class balance was near 50/50, and I spotted an early signal: pathogenic sequences had slightly different AT content than benign ones. 
I learned how to do basic quality control on biological data before touching any model.

**Day 4** — I set up the DNABERT-2 tokenizer and built a PyTorch Dataset class. 
I learned that DNA tokenization is not character-by-character but uses overlapping k-mers, and that my 513-character sequences only produce around 106 tokens, not 512. Setting max_length=512 would have wasted compute for every training step.

**Days 5-7** — I loaded DNABERT-2 and fine-tuned it. This phase involved six separate bugs: a meta-device loading error, a missing pad token, a Flash Attention CUDA crash, a Triton kernel API incompatibility, an output format mismatch, and a version conflict. 
I learned that using models with `trust_remote_code=True` requires defensive loading, and that Triton has three separate caching layers, all of which need to be cleared when you patch source files.

**Day 8** — I ran evaluation on the held-out test set. The model reached Accuracy 75.3%, F1 0.766, and ROC-AUC 0.828. 
I learned why the gap between false positives and false negatives tells you something about what the model finds harder.

**Day 9** — I tried to improve the metrics by retraining with a lower learning rate and a cosine schedule. Everything got slightly worse. 
I learned that when you have 8,000 training sequences and a 117M parameter model, the bottleneck is data, not training procedure. Knowing when to stop optimizing is its own skill.

**Days 10-11** — I built the attention extraction pipeline. Flash Attention is a fused CUDA kernel that never actually computes the attention matrix, so I had to monkey-patch the attention module to store standard PyTorch weights instead. The heatmaps revealed that early layers produce noisy attention and only layers 7-11 carry task-relevant signal. I learned the difference between what a model computes and what you can actually observe from the outside.

**Day 12** — I tried Integrated Gradients for attribution but the Triton backward kernel produced non-converging errors. I switched to Input x Gradient instead, which worked. 
I learned substituting a simpler attribution method is not failure!

**Day 13** — I trained a Random Forest on 26 nucleotide composition features (GC content, dinucleotide frequencies, CpG ratio, and others) as a classical baseline. It reached F1 0.669. DNABERT-2 reached F1 0.783. 
The gap of 0.114 is the number that actually validates the project.

**Day 14** — I wrote the scientific report and the README. :)


**The main finding** is this: a significant portion of the signal in pathogenic sequences can be captured by simple composition statistics. GC% alone gets you to F1 0.657. But the transformer consistently outperforms that ceiling because it captures positional and contextual patterns that composition features cannot see. The attention heatmaps show that the model focuses on specific nucleotide windows, not on the sequence as a whole.
