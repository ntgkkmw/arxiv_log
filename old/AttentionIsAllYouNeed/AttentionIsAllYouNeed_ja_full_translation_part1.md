# Attention Is All You Need（全訳）— Part 1/4

> **原文**: ユーザー提供のPDF「AttentionIsAllYouNeed.pdf」より（Vaswani et al.）

---

## 0. 事前表記

- **対訳形式**：英語1文 → 和訳1文（交互）
- 数式・記号（例：`d_model`, `softmax`, `O(n^2·d)`）は原文表記を優先し、必要に応じて和訳で補足します。
- 参照番号（`[1]`など）は原文のまま保持します。

---

## Permissions / 注意書き（PDF冒頭）

Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

適切な帰属表示（クレジット）を行うことを条件として、Google は本論文の表および図を、ジャーナリズムまたは学術目的の著作物に限り複製することを許可する。

---

## Title / 著者

Attention Is All You Need

注意（Attention）こそがすべてだ

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin（著者一覧）

---

## Abstract / 要旨

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.

現在主流の系列変換（sequence transduction）モデルは、エンコーダとデコーダを含む複雑な再帰型（recurrent）または畳み込み型（convolutional）のニューラルネットワークに基づいている。

The best performing models also connect the encoder and decoder through an attention mechanism.

最高性能のモデルは、さらに注意機構（attention mechanism）を介してエンコーダとデコーダを接続している。

We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

我々は、再帰や畳み込みを完全に排し、注意機構だけに基づく新しいシンプルなネットワーク構造 **Transformer** を提案する。

Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

2つの機械翻訳タスクでの実験により、本モデルは品質が優れているだけでなく、並列化しやすく、学習時間も大幅に短いことを示す。

Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

我々のモデルは WMT 2014 英独翻訳タスクで 28.4 BLEU を達成し、アンサンブルを含む既存の最良結果を 2 BLEU 以上上回った。

On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

WMT 2014 英仏翻訳タスクでは、8 GPU で 3.5 日学習するだけで 41.8 BLEU を達成し、単一モデルとして新たな最先端（SOTA）を打ち立てた（既存最良モデルの学習コストのごく一部である）。

We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

さらに、大規模データおよび限られた学習データの両方において英語構文解析（constituency parsing）へ成功裏に適用し、Transformer が他タスクにも良く汎化することを示す。

* Equal contribution.

* 同等の貢献。

Listing order is random.

著者の並び順はランダムである。

Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea.

Jakob は RNN を自己注意（self-attention）で置き換えることを提案し、その評価に向けた取り組みを開始した。

Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work.

Ashish は Illia とともに最初の Transformer モデルを設計・実装し、本研究のあらゆる側面において中核的に関与した。

Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail.

Noam はスケールド・ドット積注意（scaled dot-product attention）、マルチヘッド注意（multi-head attention）、およびパラメータ不要の位置表現を提案し、ほぼすべての詳細に関与したもう一人の中心人物となった。

Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor.

Niki はオリジナルのコードベースおよび tensor2tensor において、無数のモデル変種を設計・実装・チューニング・評価した。

Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations.

Llion も新しいモデル変種を試し、初期コードベース、効率的な推論、および可視化を担当した。

Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

Lukasz と Aidan は、以前のコードベースを置き換える tensor2tensor の各部を設計・実装し、結果を大きく改善するとともに研究を大幅に加速させた。

31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.

第31回 Neural Information Processing Systems 会議（NIPS 2017）、米国カリフォルニア州ロングビーチ。

arXiv:1706.03762v7 [cs.CL] 2 Aug 2023

arXiv:1706.03762v7 [cs.CL]（2023年8月2日版）

---

## 1 Introduction / 1 はじめに

Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5].

再帰型ニューラルネットワーク（RNN）、特に LSTM [13] やゲート付き再帰ネットワーク [7] は、言語モデルや機械翻訳といった系列モデリング／系列変換の問題における最先端手法として確立されてきた [35, 2, 5]。

Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

その後も、多くの研究が再帰型言語モデルやエンコーダ・デコーダ構造の性能限界を押し広げようとしてきた [38, 24, 15]。

Recurrent models typically factor computation along the symbol positions of the input and output sequences.

再帰型モデルは通常、入力・出力系列の各トークン位置に沿って計算を分解する。

Aligning the positions to steps in computation time, they generate a sequence of hidden states h_t, as a function of the previous hidden state h_{t−1} and the input for position t.

位置を計算時間のステップに対応づけ、前時刻の隠れ状態 h_{t−1} と位置 t の入力から、隠れ状態列 h_t を逐次生成する。

This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

この本質的に逐次的な性質は、1サンプル内での並列化を妨げるため、系列長が長くなるほど致命的になる（メモリ制約によりサンプル間のバッチ化も制限されるため）。

Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter.

近年の研究では、因数分解の工夫 [21] や条件付き計算 [32] によって計算効率を大きく改善し、後者では性能向上も得ている。

The fundamental constraint of sequential computation, however, remains.

しかし、逐次計算という根本的な制約は残ったままである。

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19].

注意機構は多様なタスクにおいて、系列モデリング／系列変換モデルの重要要素となっており、入力／出力系列内での距離に依らず依存関係を扱えるようにする [2, 19]。

In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

ただし、少数の例外 [27] を除けば、そのような注意機構は再帰ネットワークと組み合わせて用いられている。

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

本研究では、再帰を排し、入力と出力のグローバルな依存関係を注意機構のみによって捉えるモデル構造 **Transformer** を提案する。

The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

Transformer は大幅に並列化でき、P100 GPU を8枚用いてわずか12時間学習するだけで翻訳品質の新たな最先端に到達し得る。

---

## 2 Background / 2 背景

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.

逐次計算を減らすという目的は、Extended Neural GPU [16]、ByteNet [18]、ConvS2S [9] の基盤でもある。これらはいずれも畳み込みニューラルネットワークを基本ブロックとして用い、入力・出力の全位置について隠れ表現を並列に計算する。

In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet.

これらのモデルでは、任意の2位置間の信号を関連付けるのに必要な演算回数が距離とともに増加し、ConvS2S では線形、ByteNet では対数的に増える。

This makes it more difficult to learn dependencies between distant positions [12].

そのため、遠距離の依存関係を学習しづらくなる [12]。

In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

Transformer ではこれを定数回の演算に抑えられるが、注意で重み付けした位置の平均化により有効解像度が下がるという代償がある。この効果は 3.2 節で述べるマルチヘッド注意によって相殺する。

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.

自己注意（self-attention、別名 intra-attention）とは、単一系列内の異なる位置同士を関連付け、系列の表現を計算する注意機構である。

Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

自己注意は、読解（reading comprehension）、抽象要約、テキスト含意、タスク非依存の文表現学習など多様なタスクで成功している [4, 27, 28, 22]。

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

End-to-end memory network は、系列に整列した再帰の代わりに再帰的注意機構に基づいており、簡易言語の質問応答や言語モデルで良好な性能を示した [34]。

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

我々の知る限り、Transformer は、系列整列 RNN や畳み込みを用いずに、入力・出力の表現を自己注意だけで計算する初の系列変換モデルである。

In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

以降の節では、Transformer を説明し、自己注意を採用する動機を述べ、[17, 18] や [9] のようなモデルに対する利点を議論する。

---

## 3 Model Architecture（冒頭）/ 3 モデル構造（冒頭）

Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35].

競争力の高いニューラル系列変換モデルの多くは、エンコーダ・デコーダ構造を持つ [5, 2, 35]。

Here, the encoder maps an input sequence of symbol representations (x_1,...,x_n) to a sequence of continuous representations z=(z_1,...,z_n).

ここでエンコーダは、記号表現の入力系列 (x_1,...,x_n) を連続表現の系列 z=(z_1,...,z_n) に写像する。

Given z, the decoder then generates an output sequence (y_1,...,y_m) of symbols one element at a time.

z が与えられると、デコーダは出力系列 (y_1,...,y_m) を1要素ずつ生成する。

At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.

各ステップでモデルは自己回帰的（auto-regressive）[10] であり、次の記号を生成する際に、既に生成した記号を追加入力として用いる。

