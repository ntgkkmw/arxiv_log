# Attention Is All You Need（全訳）— Part 2/4

> **続き**（3 Model Architecture 以降）

---

## Figure 1（キャプション）

Figure 1: The Transformer - model architecture.

図1：Transformer のモデル構造。

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

Transformer は図1の全体構造に従い、エンコーダとデコーダの双方で、自己注意（self-attention）の積み重ねと、位置ごとの全結合層（point-wise, fully connected layers）を用いる（図1の左半分がエンコーダ、右半分がデコーダ）。



## Figure 2（キャプション）

Scaled Dot-Product Attention

スケールド・ドット積注意

Multi-Head Attention

マルチヘッド注意

Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

図2：（左）スケールド・ドット積注意。（右）マルチヘッド注意は、複数の注意層を並列に走らせたものである。


---

## 3.1 Encoder and Decoder Stacks / 3.1 エンコーダ／デコーダのスタック

Encoder: The encoder is composed of a stack of N = 6 identical layers.

エンコーダ：エンコーダは N=6 の同一層を積み重ねたスタックで構成される。

Each layer has two sub-layers.

各層は2つのサブレイヤ（下位層）を持つ。

The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network.

1つ目はマルチヘッド自己注意機構、2つ目はシンプルな「位置ごと」の全結合フィードフォワードネットワークである。

We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1].

各サブレイヤの周りに残差接続（residual connection）[11] を置き、その後にレイヤ正規化（layer normalization）[1] を適用する。

That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself.

すなわち各サブレイヤの出力は `LayerNorm(x + Sublayer(x))` であり、`Sublayer(x)` はサブレイヤ自身が実装する関数である。

To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_model = 512.

この残差接続を成立させるため、モデル内のすべてのサブレイヤと埋め込み層は、次元 `d_model=512` の出力を生成する。

Decoder: The decoder is also composed of a stack of N = 6 identical layers.

デコーダ：デコーダも同様に N=6 の同一層を積み重ねたスタックで構成される。

In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

デコーダは、各層にエンコーダ同様の2サブレイヤに加え、エンコーダスタックの出力に対してマルチヘッド注意を行う第3サブレイヤを挿入する。

Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization.

エンコーダと同様、各サブレイヤの周囲に残差接続を置き、その後にレイヤ正規化を行う。

We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions.

また、デコーダスタック内の自己注意サブレイヤは、ある位置が未来（後続）位置を参照できないように修正する。

This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

このマスキングと、出力埋め込みを1位置ずらす処理を組み合わせることで、位置 i の予測が i より前の既知出力のみに依存することを保証する。

---

## 3.2 Attention / 3.2 注意（Attention）

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

注意関数は、クエリ（query）とキー・バリュー（key-value）対の集合を入力として出力を返す写像として記述できる。ここでクエリ、キー、バリュー、出力はいずれもベクトルである。

The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

出力はバリューの重み付き和として計算され、各バリューに割り当てる重みは、クエリと対応するキーの「適合度（compatibility）」関数によって計算される。

---

### 3.2.1 Scaled Dot-Product Attention / 3.2.1 スケールド・ドット積注意

We call our particular attention "Scaled Dot-Product Attention" (Figure 2).

我々はこの注意を「スケールド・ドット積注意（Scaled Dot-Product Attention）」と呼ぶ（図2）。

The input consists of queries and keys of dimension d_k, and values of dimension d_v.

入力は、次元 `d_k` のクエリとキー、および次元 `d_v` のバリューからなる。

We compute the dot products of the query with all keys, divide each by √d_k, and apply a softmax function to obtain the weights on the values.

クエリと全キーとの内積を計算し、それぞれを `√d_k` で割ってから softmax を適用し、バリューに対する重みを得る。

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q.

実際には、複数クエリを行列 Q にまとめ、同時に注意を計算する。

The keys and values are also packed together into matrices K and V.

キーとバリューもそれぞれ行列 K, V にまとめる。

We compute the matrix of outputs as:

出力行列は次式で計算する：

Attention(Q, K, V) = softmax( Q K^T / √d_k ) V  (1)

`Attention(Q, K, V) = softmax( Q K^T / √d_k ) V` ・・・(1)

The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention.

よく使われる注意関数は、加法注意（additive attention）[2] と、ドット積（乗法）注意（dot-product / multiplicative attention）の2つである。

Dot-product attention is identical to our algorithm, except for the scaling factor of 1/√d_k.

ドット積注意は、スケーリング係数 `1/√d_k` が無い点を除けば我々のアルゴリズムと同一である。

Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.

加法注意は、単一の隠れ層を持つフィードフォワードネットワークで適合度関数を計算する。

While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

理論計算量は両者で近いが、ドット積注意は高度に最適化された行列積で実装できるため、実用上は大幅に高速かつ省メモリである。

While for small values of d_k the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_k [3].

`d_k` が小さい場合は両者の性能は似るが、`d_k` が大きい場合、スケーリング無しのドット積注意よりも加法注意の方が優れることが報告されている [3]。

We suspect that for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients.

我々は、`d_k` が大きいと内積の絶対値が大きくなり、softmax が勾配が極端に小さい領域に入ってしまうためだと推測する。

To counteract this effect, we scale the dot products by 1/√d_k.

この効果を抑えるため、内積を `1/√d_k` でスケーリングする。

To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1.

内積が大きくなる理由を示すため、q と k の各成分が平均0・分散1の独立確率変数だと仮定する。

Then their dot product, q·k = Σ_{i=1}^{d_k} q_i k_i, has mean 0 and variance d_k.

すると内積 `q·k = Σ_{i=1}^{d_k} q_i k_i` の平均は0、分散は `d_k` になる。

---

### 3.2.2 Multi-Head Attention / 3.2.2 マルチヘッド注意

Instead of performing a single attention function with d_model-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively.

`d_model` 次元のキー・バリュー・クエリで単一の注意を行う代わりに、クエリ・キー・バリューをそれぞれ学習可能な線形射影で h 回（異なる射影）変換し、次元をそれぞれ `d_k`, `d_k`, `d_v` に落とすことが有益だと分かった。

On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding d_v-dimensional output values.

射影した各（クエリ・キー・バリュー）の組に対して注意を並列に計算し、`d_v` 次元の出力を得る。

These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

これらの出力を連結（concat）し、さらにもう一度射影して最終出力を得る（図2）。

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.

マルチヘッド注意により、モデルは異なる表現部分空間（representation subspaces）の情報に、異なる位置で同時に注意を向けられる。

With a single attention head, averaging inhibits this.

単一ヘッドの場合、平均化によってこの性質が抑制されてしまう。

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

`MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O`

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).

ここで `head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)` である。

Where the projections are parameter matrices W_i^Q ∈ R^{d_model×d_k}, W_i^K ∈ R^{d_model×d_k}, W_i^V ∈ R^{d_model×d_v} and W^O ∈ R^{h d_v×d_model}.

射影はパラメータ行列 `W_i^Q ∈ R^{d_model×d_k}`, `W_i^K ∈ R^{d_model×d_k}`, `W_i^V ∈ R^{d_model×d_v}`, `W^O ∈ R^{h d_v×d_model}` で表される。

In this work we employ h = 8 parallel attention layers, or heads.

本研究では `h=8` の並列注意層（ヘッド）を用いる。

For each of these we use d_k = d_v = d_model / h = 64.

各ヘッドは `d_k=d_v=d_model/h=64` とする。

Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

各ヘッドの次元を小さくしているため、全体の計算コストはフル次元の単一ヘッド注意と同程度になる。

---

### 3.2.3 Applications of Attention in our Model / 3.2.3 本モデルにおける注意の使い分け

The Transformer uses multi-head attention in three different ways:

Transformer はマルチヘッド注意を3通りの方法で用いる：

In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder.

「エンコーダ・デコーダ注意」層では、クエリは直前のデコーダ層から来て、メモリのキーとバリューはエンコーダ出力から来る。

This allows every position in the decoder to attend over all positions in the input sequence.

これによりデコーダの各位置は、入力系列の全位置に注意を向けられる。

This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].

これは [38, 2, 9] のような sequence-to-sequence モデルにおける典型的なエンコーダ・デコーダ注意を模倣している。

The encoder contains self-attention layers.

エンコーダには自己注意層が含まれる。

In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder.

自己注意層では、キー・バリュー・クエリはすべて同じ場所、ここではエンコーダの前層出力から来る。

Each position in the encoder can attend to all positions in the previous layer of the encoder.

エンコーダの各位置は、前層の全位置に注意を向けられる。

Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position.

同様に、デコーダの自己注意層では、各位置がその位置まで（自身を含む）のデコーダ位置すべてに注意を向けられる。

We need to prevent leftward information flow in the decoder to preserve the auto-regressive property.

自己回帰性を保つため、デコーダでは未来方向（左から右生成に対する「右側」）の情報流入を防ぐ必要がある。

We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections.

これはスケールド・ドット積注意の内部で、softmax 入力のうち不正な接続に対応する値をマスク（−∞ に設定）することで実現する。

See Figure 2.

図2を参照。

---

## 3.3 Position-wise Feed-Forward Networks / 3.3 位置ごとのフィードフォワードネットワーク

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.

注意サブレイヤに加えて、エンコーダとデコーダの各層には全結合のフィードフォワードネットワークが含まれ、各位置に独立かつ同一の形で適用される。

This consists of two linear transformations with a ReLU activation in between.

これは2つの線形変換の間に ReLU 活性化を挟んだものからなる。

FFN(x) = max(0, x W_1 + b_1) W_2 + b_2  (2)

`FFN(x) = max(0, x W_1 + b_1) W_2 + b_2` ・・・(2)

While the linear transformations are the same across different positions, they use different parameters from layer to layer.

線形変換は位置間で同一だが、層ごとには異なるパラメータを用いる。

Another way of describing this is as two convolutions with kernel size 1.

別の見方をすれば、カーネルサイズ1の畳み込みを2回行うのと同等である。

The dimensionality of input and output is d_model = 512, and the inner-layer has dimensionality d_ff = 2048.

入出力次元は `d_model=512`、中間層の次元は `d_ff=2048` である。

---

## 3.4 Embeddings and Softmax / 3.4 埋め込みと Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_model.

他の系列変換モデルと同様、入力トークンと出力トークンを `d_model` 次元ベクトルに変換する学習可能な埋め込み（embedding）を用いる。

We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

また、一般的な学習可能線形変換と softmax により、デコーダ出力を次トークン確率に変換する。

In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30].

本モデルでは [30] と同様に、2つの埋め込み層と pre-softmax の線形変換で同じ重み行列を共有する。

In the embedding layers, we multiply those weights by √d_model.

埋め込み層では、その重みを `√d_model` でスケールする。

---

## 3.5 Positional Encoding / 3.5 位置エンコーディング

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.

本モデルには再帰も畳み込みもないため、系列の順序情報を利用するには、トークンの相対／絶対位置に関する情報を注入する必要がある。

To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

このため、エンコーダ／デコーダスタックの最下部で、入力埋め込みに「位置エンコーディング（positional encodings）」を加算する。

The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.

位置エンコーディングは埋め込みと同じ次元 `d_model` を持ち、両者を足し合わせられる。

There are many choices of positional encodings, learned and fixed [9].

位置エンコーディングには学習型・固定型など多くの選択肢がある [9]。

In this work, we use sine and cosine functions of different frequencies:

本研究では周波数の異なる正弦・余弦関数を用いる：

PE(pos, 2i) = sin(pos / 10000^{2i/d_model})

`PE(pos, 2i) = sin(pos / 10000^{2i/d_model})`

PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

`PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})`

where pos is the position and i is the dimension.

ここで `pos` は位置、`i` は次元である。

That is, each dimension of the positional encoding corresponds to a sinusoid.

つまり位置エンコーディングの各次元は1つの正弦波に対応する。

The wavelengths form a geometric progression from 2π to 10000·2π.

波長は 2π から 10000·2π までの等比級数（幾何級数）となる。

We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE_{pos+k} can be represented as a linear function of PE_pos.

この関数を選んだのは、任意の固定オフセット k に対して `PE_{pos+k}` が `PE_pos` の線形関数として表せるため、相対位置に基づいて注意を向けることをモデルが容易に学習できると仮定したからである。

We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)).

学習可能な位置埋め込み [9] も試したが、両者はほぼ同一の結果を示した（表3の行(E)参照）。

We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

正弦波版を選んだのは、学習時に見たより長い系列長にも外挿できる可能性があるためである。
