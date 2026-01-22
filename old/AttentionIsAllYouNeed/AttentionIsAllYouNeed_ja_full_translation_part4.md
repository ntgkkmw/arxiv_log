# Attention Is All You Need（全訳）— Part 4/4

> **続き**（Acknowledgements / References / Appendix figures）

---

## Acknowledgements / 謝辞

We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

Nal Kalchbrenner と Stephan Gouws に、有益なコメント、修正、そして着想を与えてくれたことに感謝する。

---

## References / 参考文献

> 参考文献は原文を保持し、必要最小限の補足のみ和訳します（タイトル等）。

[1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[1] Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E Hinton. Layer normalization（レイヤ正規化）. arXiv:1607.06450, 2016.

[2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. CoRR, abs/1409.0473, 2014.

[2] Bahdanau, Cho, Bengio. アラインメントと翻訳を同時に学習するニューラル機械翻訳. CoRR:1409.0473, 2014.

[3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.

[3] Britz ら. ニューラル機械翻訳アーキテクチャの大規模探索. CoRR:1703.03906, 2017.

[4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine reading. arXiv preprint arXiv:1601.06733, 2016.

[4] Cheng ら. 機械読解のための LSTM ネットワーク. arXiv:1601.06733, 2016.

[5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

[5] Cho ら. 統計的機械翻訳のための RNN エンコーダ・デコーダによる句表現学習. CoRR:1406.1078, 2014.

[6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv preprint arXiv:1610.02357, 2016.

[6] Chollet. Xception：Depthwise separable convolution による深層学習. arXiv:1610.02357, 2016.

[7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

[7] Chung ら. 系列モデリングにおけるゲート付き再帰ネットの実証評価. CoRR:1412.3555, 2014.

[8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural network grammars. In Proc. of NAACL, 2016.

[8] Dyer ら. RNN Grammar. NAACL, 2016.

[9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

[9] Gehring ら. 畳み込みによる seq2seq 学習. arXiv:1705.03122v2, 2017.

[10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

[10] Graves. RNN による系列生成. arXiv:1308.0850, 2013.

[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, 2016.

[11] He ら. 画像認識のための深層残差学習. CVPR, 2016.

[12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[12] Hochreiter ら. 再帰ネットにおける勾配流：長期依存の学習困難性, 2001.

[13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 1997.

[13] Hochreiter & Schmidhuber. LSTM. Neural Computation, 1997.

[14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations across languages. EMNLP, 2009.

[14] Huang & Harper. 潜在注釈を用いた PCFG の自己学習. EMNLP, 2009.

[15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

[15] Jozefowicz ら. 言語モデルの限界探索. arXiv:1602.02410, 2016.

[16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In NIPS, 2016.

[16] Kaiser & Bengio. Active memory は attention を置き換えられるか？ NIPS, 2016.

[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. ICLR, 2016.

[17] Kaiser & Sutskever. Neural GPU はアルゴリズムを学習する. ICLR, 2016.

[18] Nal Kalchbrenner et al. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

[18] Kalchbrenner ら. 線形時間のニューラル機械翻訳. arXiv:1610.10099v2, 2017.

[19] Yoon Kim et al. Structured attention networks. ICLR, 2017.

[19] Kim ら. Structured attention network. ICLR, 2017.

[20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. ICLR, 2015.

[20] Kingma & Ba. Adam. ICLR, 2015.

[21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.

[21] Kuchaiev & Ginsburg. LSTM の因数分解テクニック. arXiv:1703.10722, 2017.

[22] Zhouhan Lin et al. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

[22] Lin ら. 構造化 self-attentive 文埋め込み. arXiv:1703.03130, 2017.

[23] Minh-Thang Luong et al. Multi-task sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.

[23] Luong ら. マルチタスク seq2seq 学習. arXiv:1511.06114, 2015.

[24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.

[24] Luong ら. attention ベース NMT の効果的アプローチ. arXiv:1508.04025, 2015.

[25] Mitchell P Marcus et al. Building a large annotated corpus of English: the Penn Treebank. Computational Linguistics, 1993.

[25] Marcus ら. Penn Treebank. Computational Linguistics, 1993.

[26] David McClosky et al. Effective self-training for parsing. NAACL-HLT, 2006.

[26] McClosky ら. 解析のための効果的自己学習. NAACL-HLT, 2006.

[27] Ankur Parikh et al. A decomposable attention model. EMNLP, 2016.

[27] Parikh ら. 分解可能 attention モデル. EMNLP, 2016.

[28] Romain Paulus et al. A deep reinforced model for abstractive summarization. arXiv preprint arXiv:1705.04304, 2017.

[28] Paulus ら. 抽象要約のための深層強化モデル. arXiv:1705.04304, 2017.

[29] Slav Petrov et al. Learning accurate, compact, and interpretable tree annotation. ACL, 2006.

[29] Petrov ら. 正確・コンパクト・解釈可能な木注釈の学習. ACL, 2006.

[30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv preprint arXiv:1608.05859, 2016.

[30] Press & Wolf. 出力埋め込みで言語モデル改善. arXiv:1608.05859, 2016.

[31] Rico Sennrich et al. Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909, 2015.

[31] Sennrich ら. サブワードによるレア語 NMT. arXiv:1508.07909, 2015.

[32] Noam Shazeer et al. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

[32] Shazeer ら. Sparse-gated MoE 層. arXiv:1701.06538, 2017.

[33] Nitish Srivastava et al. Dropout: a simple way to prevent neural networks from overfitting. JMLR, 2014.

[33] Srivastava ら. Dropout. JMLR, 2014.

[34] Sainbayar Sukhbaatar et al. End-to-end memory networks. NIPS, 2015.

[34] Sukhbaatar ら. End-to-end memory network. NIPS, 2015.

[35] Ilya Sutskever et al. Sequence to sequence learning with neural networks. NIPS, 2014.

[35] Sutskever ら. seq2seq 学習. NIPS, 2014.

[36] Christian Szegedy et al. Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

[36] Szegedy ら. Inception の再考（ラベルスムージング等）. CoRR:1512.00567, 2015.

[37] Vinyals & Kaiser et al. Grammar as a foreign language. NIPS, 2015.

[37] Vinyals & Kaiser ら. 文法を外国語として扱う. NIPS, 2015.

[38] Yonghui Wu et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

[38] Wu ら. Google NMT. arXiv:1609.08144, 2016.

[39] Jie Zhou et al. Deep recurrent models with fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

[39] Zhou ら. Fast-forward 接続付き深層再帰 NMT. CoRR:1606.04199, 2016.

[40] Muhua Zhu et al. Fast and accurate shift-reduce constituent parsing. ACL, 2013.

[40] Zhu ら. 高速高精度 shift-reduce 句構造解析. ACL, 2013.

---

## Appendix: Attention Visualizations / 付録：注意の可視化（図3〜5）

Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6.

図3：6層中5層目のエンコーダ自己注意において、注意機構が長距離依存を追跡している例。

Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’.

多くの注意ヘッドが動詞「making」の遠距離依存に注意を向け、「making ... more difficult」という句を完成させている。

Attentions here shown only for the word ‘making’.

ここでの注意分布は単語「making」に対してのみ表示している。

Different colors represent different heads.

色の違いはヘッドの違いを表す。

Best viewed in color.

カラー表示で見るのが望ましい。

Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution.

図4：同じく6層中5層目にある2つの注意ヘッドで、照応解析（anaphora resolution）に関与しているように見える。

Top: Full attentions for head 5.

上：ヘッド5の全注意分布。

Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6.

下：単語「its」からの注意のみを、ヘッド5とヘッド6について抽出したもの。

Note that the attentions are very sharp for this word.

この単語に対する注意が非常に鋭い（集中している）ことに注目されたい。

Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence.

図5：多くの注意ヘッドが、文の構造に関連するように見える振る舞いを示す。

We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6.

上にはそのような例を2つ示した。いずれも6層中5層目のエンコーダ自己注意の異なるヘッドからの例である。

The heads clearly learned to perform different tasks.

ヘッドが異なるタスクを学習していることが明確に分かる。
