# Attention Is All You Need（全訳）— Part 3/4

> **続き**（4 Why Self-Attention ～ 6 Results）

---

## 4 Why Self-Attention / 4 なぜ自己注意か

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x_1,...,x_n) to another sequence of equal length (z_1,...,z_n), with x_i, z_i ∈ R^d, such as a hidden layer in a typical sequence transduction encoder or decoder.

本節では、自己注意層のさまざまな側面を、一般的な系列変換エンコーダ／デコーダの隠れ層としてよく用いられる再帰層や畳み込み層と比較する。ここで、可変長の記号表現系列 (x_1,...,x_n) を同じ長さの系列 (z_1,...,z_n)（x_i, z_i ∈ R^d）へ写像する状況を想定する。

Motivating our use of self-attention we consider three desiderata.

自己注意を採用する動機として、我々は3つの要件（desiderata）を考える。

One is the total computational complexity per layer.

1つ目は、層あたりの総計算量である。

Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

2つ目は、並列化できる計算量であり、必要な逐次演算の最小回数で測る。

The third is the path length between long-range dependencies in the network.

3つ目は、ネットワーク内の長距離依存（long-range dependencies）間のパス長（path length）である。

Learning long-range dependencies is a key challenge in many sequence transduction tasks.

長距離依存の学習は、多くの系列変換タスクにおける主要な課題である。

One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network.

依存関係を学習できるかどうかに影響する重要要因の一つは、前向き／後ろ向きの信号がネットワーク内を伝播する際に通る経路の長さである。

The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12].

入力・出力系列内の任意の位置の組み合わせ間でパスが短いほど、長距離依存は学習しやすい [12]。

Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

そこで我々は、異なる層タイプで構成されたネットワークにおいて、任意の2位置間の最大パス長も比較する。



### Table 1（表1）

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.

表1：層タイプごとの最大パス長、層あたり計算量、逐次演算の最小回数。n は系列長、d は表現次元、k は畳み込みのカーネルサイズ、r は制限付き自己注意における近傍サイズである。

| Layer Type（層タイプ） | Complexity per Layer（層あたり計算量） | Sequential Operations（逐次演算） | Maximum Path Length（最大パス長） |
|---|---:|---:|---:|
| Self-Attention（自己注意） | O(n^2 · d) | O(1) | O(1) |
| Recurrent（再帰） | O(n · d^2) | O(n) | O(n) |
| Convolutional（畳み込み） | O(k · n · d^2) | O(1) | O(log_k(n)) |
| Self-Attention (restricted)（自己注意：制限付き） | O(r · n · d) | O(1) | O(n/r) |


As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations.

表1に示す通り、自己注意層は定数回の逐次演算で全位置を接続できる一方、再帰層は O(n) 回の逐次演算を要する。

In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations.

計算量の観点では、系列長 n が表現次元 d より小さい場合、自己注意層は再帰層より高速になり得る。これは word-piece [38] や byte-pair [31] のような表現を用いる最先端機械翻訳モデルで、多くの場合に成り立つ。

To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position.

非常に長い系列を扱うタスクでは計算性能を改善するため、自己注意を「出力位置の周辺 r 近傍」に制限することもできる。

This would increase the maximum path length to O(n/r).

その場合、最大パス長は O(n/r) に増加する。

We plan to investigate this approach further in future work.

このアプローチは今後の研究でさらに調査する予定である。

A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions.

カーネル幅 k < n の単一畳み込み層では、入力・出力の全ての位置ペアを接続できない。

Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(log_k(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network.

全接続を実現するには、連続カーネルなら O(n/k) 層、拡張（dilated）畳み込みなら O(log_k(n)) 層が必要となり [18]、結果としてネットワーク内の任意2位置間の最長パスが長くなる。

Convolutional layers are generally more expensive than recurrent layers, by a factor of k.

畳み込み層は一般に、再帰層よりも k 倍程度コストが高い。

Separable convolutions [6], however, decrease the complexity considerably, to O(k·n·d + n·d^2).

ただし、分離畳み込み（separable convolutions）[6] は計算量を O(k·n·d + n·d^2) まで大きく低減する。

Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

それでも k=n の場合、分離畳み込みの計算量は、自己注意層と位置ごとのフィードフォワード層を組み合わせた計算量と等しくなる。これは我々のモデルが採用するアプローチである。

As side benefit, self-attention could yield more interpretable models.

副次的利点として、自己注意はより解釈可能なモデルにつながり得る。

We inspect attention distributions from our models and present and discuss examples in the appendix.

我々はモデルの注意分布を調べ、付録で例を提示し議論する。

Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

個々の注意ヘッドが異なるタスクを学習していることが明確に見て取れるだけでなく、多くのヘッドが文の統語・意味構造に関連する振る舞いを示すように見える。

---

## 5 Training / 5 学習

This section describes the training regime for our models.

本節ではモデルの学習手順を説明する。

### 5.1 Training Data and Batching / 5.1 学習データとバッチング

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs.

標準的な WMT 2014 英独データセット（約450万文対）で学習した。

Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens.

文は byte-pair encoding（BPE）[3] で符号化し、ソース・ターゲットで共有する語彙は約37,000トークンである。

For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38].

英仏では、より大規模な WMT 2014 英仏データセット（3,600万文）を用い、トークンを 32,000 の word-piece 語彙に分割した [38]。

Sentence pairs were batched together by approximate sequence length.

文対はおおよその系列長でまとめてバッチ化した。

Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

各学習バッチは、およそ 25,000 のソーストークンと 25,000 のターゲットトークンを含む文対の集合からなる。

### 5.2 Hardware and Schedule / 5.2 ハードウェアとスケジュール

We trained our models on one machine with 8 NVIDIA P100 GPUs.

モデルは NVIDIA P100 GPU を8枚搭載した1台のマシンで学習した。

For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds.

本論文で説明するハイパーパラメータを用いた base モデルでは、学習ステップあたり約0.4秒かかった。

We trained the base models for a total of 100,000 steps or 12 hours.

base モデルは合計 100,000 ステップ、すなわち12時間学習した。

For our big models (described on the bottom line of table 3), step time was 1.0 seconds.

big モデル（表3の最下行）では、ステップあたり1.0秒であった。

The big models were trained for 300,000 steps (3.5 days).

big モデルは 300,000 ステップ（3.5日）学習した。

### 5.3 Optimizer / 5.3 オプティマイザ

We used the Adam optimizer [20] with β_1 = 0.9, β_2 = 0.98 and ϵ = 10^{−9}.

Adam オプティマイザ [20] を用い、β_1=0.9、β_2=0.98、ϵ=10^{-9} とした。

We varied the learning rate over the course of training, according to the formula:

学習率は学習の進行に応じて次式で変化させた：

lrate = d_model^{−0.5} · min(step_num^{−0.5}, step_num · warmup_steps^{−1.5})  (3)

`lrate = d_model^{−0.5} · min(step_num^{−0.5}, step_num · warmup_steps^{−1.5})` ・・・(3)

This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number.

これは最初の warmup_steps ステップは学習率を線形に増加させ、その後はステップ数の逆平方根に比例して減少させることに対応する。

We used warmup_steps = 4000.

warmup_steps=4000 とした。

### 5.4 Regularization / 5.4 正則化

We employ three types of regularization during training:

Attention Dropout.

注意（Attention）ドロップアウト。

We apply dropout to the attention weights (the output of the softmax) in the attention mechanism.

注意機構において、注意重み（softmax の出力）に dropout を適用する。

学習中に3種類の正則化を用いる：

Residual Dropout.

残差ドロップアウト。

We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized.

各サブレイヤの出力に dropout [33] を適用し、その後サブレイヤ入力に加算して正規化する前に行う。

In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.

さらに、エンコーダとデコーダの両スタックで、埋め込みと位置エンコーディングの和に対しても dropout を適用する。

For the base model, we use a rate of P_drop = 0.1.

base モデルでは dropout 率 P_drop=0.1 を用いる。

Label Smoothing.

ラベルスムージング。

During training, we employed label smoothing of value ϵ_ls = 0.1 [36].

学習中に ϵ_ls=0.1 のラベルスムージング [36] を用いた。

This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

これはモデルがより不確かになるため perplexity は悪化するが、精度と BLEU スコアは改善する。

---

## 6 Results / 6 結果

### 6.1 Machine Translation / 6.1 機械翻訳

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4.

WMT 2014 英独翻訳タスクでは、big Transformer（表2の Transformer (big)）が、アンサンブルを含む既報の最良モデルを 2.0 BLEU 以上上回り、28.4 BLEU の新しい SOTA を確立した。

The configuration of this model is listed in the bottom line of Table 3.

このモデルの設定は表3の最下行に示す。

Training took 3.5 days on 8 P100 GPUs.

学習には P100 GPU 8枚で 3.5 日かかった。

Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

base モデルでさえ、既報のすべてのモデルおよびアンサンブルを上回り、学習コストは競合モデルのごく一部である。

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model.

WMT 2014 英仏翻訳タスクでは、big モデルが 41.0 BLEU を達成し、既報の単一モデルをすべて上回った。学習コストは従来 SOTA モデルの 1/4 未満である。

The Transformer (big) model trained for English-to-French used dropout rate P_drop = 0.1, instead of 0.3.

英仏用に学習した Transformer (big) は、0.3 ではなく P_drop=0.1 を用いた。

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals.

base モデルでは、10分間隔で保存した最後の5つのチェックポイントを平均した単一モデルを用いた。

For the big models, we averaged the last 20 checkpoints.

big モデルでは、最後の20チェックポイントを平均した。

We used beam search with a beam size of 4 and length penalty α = 0.6 [38].

ビーム幅4、長さペナルティ α=0.6 のビームサーチを用いた [38]。

These hyperparameters were chosen after experimentation on the development set.

これらのハイパーパラメータは開発セットでの実験を経て選択した。

We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

推論時の最大出力長は入力長+50とし、可能な場合は早期終了する [38]。



### Table 2（表2）

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

表2：Transformer は、英独・英仏の newstest2014 テストにおいて、従来の SOTA を学習コストのごく一部で上回る BLEU を達成する。

| Model（モデル） | BLEU EN-DE | BLEU EN-FR | Training Cost (FLOPs) EN-DE | Training Cost (FLOPs) EN-FR |
|---|---:|---:|---:|---:|
| ByteNet [18] | 23.75 | — | — | — |
| Deep-Att + PosUnk [39] | — | 39.2 | 1.0·10^20 | — |
| GNMT + RL [38] | 24.6 | 39.92 | 2.3·10^19 | 1.4·10^20 |
| ConvS2S [9] | 25.16 | 40.46 | 9.6·10^18 | 1.5·10^20 |
| MoE [32] | 26.03 | 40.56 | 2.0·10^19 | 1.2·10^20 |
| Deep-Att + PosUnk Ensemble [39] | — | 40.4 | 8.0·10^20 | — |
| GNMT + RL Ensemble [38] | 26.30 | 41.16 | 1.8·10^20 | 1.1·10^21 |
| ConvS2S Ensemble [9] | 26.36 | 41.29 | 7.7·10^19 | 1.2·10^21 |
| Transformer (base model) | 27.3 | 38.1 | 3.3·10^18 | — |
| Transformer (big) | 28.4 | 41.8 | 2.3·10^19 | — |

※「—」は原文表に値が掲載されていない（または該当しない）ことを示す。


Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature.

表2は結果をまとめ、翻訳品質と学習コストを既存の他のモデル構造と比較している。

We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU.

モデル学習で用いた浮動小数点演算回数は、学習時間、GPU枚数、および各GPUの単精度演算性能（持続値）の推定値を掛け合わせて見積もった。

We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.

K80, K40, M40, P100 の持続性能として、それぞれ 2.8, 3.7, 6.0, 9.5 TFLOPS を用いた。

### 6.2 Model Variations / 6.2 モデル変種

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013.

Transformer の各構成要素の重要性を評価するため、base モデルを様々に変更し、英独翻訳の開発セット newstest2013 における性能変化を測定した。

We used beam search as described in the previous section, but no checkpoint averaging.

前節で述べたビームサーチを用いるが、チェックポイント平均は行わない。

We present these results in Table 3.


### Table 3（表3）

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

表3：Transformer アーキテクチャの変種。記載のない値は base モデルと同一。指標はすべて英独翻訳の開発セット newstest2013 上。perplexity は BPE の wordpiece あたりであり、単語あたり perplexity と比較すべきではない。

> **注**：PDFの表は複数行にわたり「base から変更したパラメータだけ」を行ごとに列挙する形式になっています。本訳でも、その表記に合わせて **「変化したパラメータ → 結果」** を忠実に再掲します（base と同一の列は省略）。

**base**

- base: N=6, d_model=512, d_ff=2048, h=8, d_k=64, d_v=64, P_drop=0.1, ε_ls=0.1, train steps=100K, PPL(dev)=4.92, BLEU(dev)=25.8, params×10^6=65

**(A) ヘッド数 h と d_k/d_v（計算量一定）**

- h=1, d_k=512, d_v=512 → PPL=5.29, BLEU=24.9
- h=4, d_k=128, d_v=128 → PPL=5.00, BLEU=25.5
- h=16, d_k=32, d_v=32 → PPL=4.91, BLEU=25.8
- h=32, d_k=16, d_v=16 → PPL=5.01, BLEU=25.4

**(B) 注意キーサイズ d_k の縮小**

- d_k=16 → PPL=5.16, BLEU=25.1, params×10^6=58
- d_k=32 → PPL=5.01, BLEU=25.4, params×10^6=60

**(C) モデル規模の変更（層数や次元など）**

- N=2 → PPL=6.11, BLEU=23.7, params×10^6=36
- N=4 → PPL=5.19, BLEU=25.3, params×10^6=50
- N=8 → PPL=4.88, BLEU=25.5, params×10^6=80
- d_model=256, d_k=32, d_v=32 → PPL=5.75, BLEU=24.5, params×10^6=28
- d_model=1024, d_k=128, d_v=128 → PPL=4.66, BLEU=26.0, params×10^6=168
- d_ff=1024 → PPL=5.12, BLEU=25.4, params×10^6=53
- d_ff=4096 → PPL=4.75, BLEU=26.2, params×10^6=90

**(D) Dropout と Label Smoothing**

- P_drop=0.0 → PPL=5.77, BLEU=24.6
- P_drop=0.2 → PPL=4.95, BLEU=25.5
- ε_ls=0.0 → PPL=4.67, BLEU=25.3
- ε_ls=0.2 → PPL=5.47, BLEU=25.7

**(E) 位置表現の差し替え**

- learned positional embedding instead of sinusoids → PPL=4.92, BLEU=25.7

**big**

- big: N=6, d_model=1024, d_ff=4096, h=16, P_drop=0.3, train steps=300K, PPL(dev)=4.33, BLEU(dev)=26.4, params×10^6=213



### Table 4（表4）

Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)

表4：Transformer は英語句構造解析に良く汎化する（結果は WSJ Section 23）。

| Parser（解析器） | Training（学習設定） | WSJ 23 F1 |
|---|---|---:|
| Vinyals & Kaiser et al. (2014) [37] | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) [29] | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) [40] | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) [8] | WSJ only, discriminative | 91.7 |
| Transformer (4 layers) | WSJ only, discriminative | 91.3 |
| Zhu et al. (2013) [40] | semi-supervised | 91.3 |
| Huang & Harper (2009) [14] | semi-supervised | 91.3 |
| McClosky et al. (2006) [26] | semi-supervised | 92.1 |
| Vinyals & Kaiser et al. (2014) [37] | semi-supervised | 92.1 |
| Transformer (4 layers) | semi-supervised | 92.7 |
| Luong et al. (2015) [23] | multi-task | 93.0 |
| Dyer et al. (2016) [8] | generative | 93.3 |


Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

表4の結果は、タスク特化のチューニングがないにもかかわらず本モデルが驚くほど良く動作し、Recurrent Neural Network Grammar [8] を除く既報の全モデルより良い結果を示すことを示している。

In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the BerkeleyParser [29] even when training only on the WSJ training set of 40K sentences.

RNN の seq2seq モデル [37] と対照的に、Transformer は WSJ の 4万文だけで学習しても BerkeleyParser [29] を上回る。

---

## 7 Conclusion / 7 結論

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

本研究では Transformer を提示した。これは注意だけに基づく初の系列変換モデルであり、エンコーダ・デコーダ構造で一般的だった再帰層をマルチヘッド自己注意で置き換える。

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.

翻訳タスクでは、Transformer は再帰や畳み込みに基づく構造より大幅に高速に学習できる。

On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art.

WMT 2014 英独および英仏翻訳タスクの両方で、新しい最先端（SOTA）を達成した。

In the former task our best model outperforms even all previously reported ensembles.

前者（英独）では、最良モデルが既報のすべてのアンサンブルをも上回った。

We are excited about the future of attention-based models and plan to apply them to other tasks.

注意ベースのモデルの将来に期待しており、他タスクへ適用する予定である。

We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video.

テキスト以外の入出力モダリティを含む問題へ Transformer を拡張し、画像・音声・動画のような大規模入出力を効率的に扱うため、局所的で制限付きの注意機構も調査する計画である。

Making generation less sequential is another research goals of ours.

生成をより非逐次的にすることも、我々の研究目標の一つである。

The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.

学習と評価に用いたコードは https://github.com/tensorflow/tensor2tensor で公開している。
