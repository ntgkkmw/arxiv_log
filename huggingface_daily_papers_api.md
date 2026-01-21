# Hugging Face `GET /api/daily_papers` 調査メモ（リクエストパラメータ & レスポンス）

> 対象: `https://huggingface.co/api/daily_papers` citeturn1search5  
> 目的: このエンドポイントに **リクエストできるパラメータ** と **返ってくるデータ（レスポンススキーマ）** を把握する。

---

## 1. 概要

- `GET /api/daily_papers` は、Hugging Face の「Daily Papers」機能（キュレーションされた論文一覧）を返す公開 API エンドポイントです。citeturn1search7  
- 返却形式は JSON です。citeturn1search7  
- 認証は不要（少なくとも読み取り用途では不要）とされています。citeturn1search7  
- Hugging Face Hub 全体の API はレート制限の対象であり、OpenAPI 仕様（`.well-known/openapi.json`）が公開されています。citeturn1search18  

---

## 2. エンドポイント

- **Method**: `GET` citeturn1search7  
- **URL**: `https://huggingface.co/api/daily_papers` citeturn1search5  
- **Content-Type**: `application/json`（レスポンス）citeturn1search7  

---

## 3. リクエスト（Query Parameters）

`/api/daily_papers` は日付指定とページングに対応します。citeturn1search7  

### 3.1 `date`

- **名前**: `date` citeturn1search7  
- **型**: `string` citeturn1search7  
- **形式**: `YYYY-MM-DD` citeturn1search7  
- **必須**: 非公式ドキュメントでは「必須」と記載があります。citeturn1search7  
- **備考（挙動の観測）**: 実際には `date` を付けずに呼び出しても JSON 配列が返る例が確認できます（＝デフォルトで当日分が返る可能性）。citeturn1search5  

### 3.2 `page`

- **名前**: `page` citeturn1search7  
- **型**: `integer` citeturn1search7  
- **必須**: 任意 citeturn1search7  
- **意味**: ページ番号（`1` から開始）。citeturn1search7  

### 3.3 `limit`

- **名前**: `limit` citeturn1search7  
- **型**: `integer` citeturn1search7  
- **必須**: 任意 citeturn1search7  
- **意味**: 1ページあたりの件数。citeturn1search7  
- **デフォルト（非公式記載）**: `10–20` 程度とされています（実値は環境/時期で変わる可能性）。citeturn1search7  

---

## 4. レスポンス（Response Body）

### 4.1 レスポンス全体

- **トップレベル**: JSON の **配列**（`Array`）が返ります。citeturn1search5  
- **配列要素**: 各要素は「Daily Papers の1件（キュレーション枠）」を表すオブジェクトです。citeturn1search5  

> 参考: 実際のレスポンスは `[{ ... }, { ... }, ...]` という形で返っています。citeturn1search5


### 4.2 配列要素（DailyPaperItem）— 観測できた主なフィールド

以下は、`/api/daily_papers` の実レスポンス例から観測できたフィールドです。citeturn1search5  

#### (A) アイテム直下

- `paper` : `object` — 論文本体のメタ情報。citeturn1search5  
- `publishedAt` : `string`（ISO 8601 風）— Daily Papers 側での表示/収録に関する日時（例: `2025-02-21T21:19:35.358Z`）。citeturn1search5  
- `title` : `string` — Daily Papers 側の表示タイトル（多くの場合 `paper.title` と同一）。citeturn1search5  
- `thumbnail` : `string`（URL）— SNS サムネイル画像 URL。citeturn1search5  
- `numComments` : `integer` — コメント数。citeturn1search5  
- `submittedBy` : `object` — 投稿者（推薦者）情報。citeturn1search5  
- `isAuthorParticipating` : `boolean` — 著者が議論に参加しているかのフラグ。citeturn1search5  
- `mediaUrls` : `array[string]`（任意）— 添付メディア URL の配列（存在する例あり）。citeturn1search5  

#### (B) `paper` オブジェクト

- `paper.id` : `string` — 論文ID（arXiv ID 形式の値が入る例あり）。citeturn1search5  
- `paper.authors` : `array[object]` — 著者配列。citeturn1search5  
- `paper.publishedAt` : `string` — 論文公開日時（例: `2025-02-16T20:33:59.000Z`）。citeturn1search5  
- `paper.title` : `string` — 論文タイトル。citeturn1search5  
- `paper.summary` : `string` — 要約（abstract 相当）。citeturn1search5  
- `paper.upvotes` : `integer` — upvote 数。citeturn1search5  
- `paper.discussionId` : `string` — 議論スレッドのID。citeturn1search5  

#### (C) `paper.authors[]`（著者要素）

- `_id` : `string` — 内部ID。citeturn1search5  
- `name` : `string` — 著者名。citeturn1search5  
- `hidden` : `boolean` — 非表示フラグ（存在する例あり）。citeturn1search5  
- `status` : `string`（任意）— 例: `claimed_verified`, `extracted_confirmed` 等。citeturn1search5  
- `statusLastChangedAt` : `string`（任意）— `status` 更新日時。citeturn1search5  
- `user` : `object`（任意）— Hugging Face アカウントと紐づく場合のユーザー情報。citeturn1search5  

#### (D) `paper.authors[].user`（著者の HF ユーザー情報）

- `_id` : `string` citeturn1search5  
- `avatarUrl` : `string` citeturn1search5  
- `fullname` : `string` citeturn1search5  
- `user` : `string`（ユーザー名）citeturn1search5  
- `type` : `string`（例: `user`）citeturn1search5  
- `isPro` : `boolean` citeturn1search5  

#### (E) `submittedBy`（投稿者情報）

- `_id` : `string` citeturn1search5  
- `avatarUrl` : `string` citeturn1search5  
- `fullname` : `string` citeturn1search5  
- `name` : `string`（ユーザー名）citeturn1search5  
- `type` : `string`（例: `user`）citeturn1search5  
- `isPro` : `boolean` citeturn1search5  
- `isHf` : `boolean`（HF関係者フラグのような値が入る例あり）citeturn1search5  
- `isMod` : `boolean` citeturn1search5  
- `followerCount` : `integer`（存在する例あり）citeturn1search5  

---

## 5. サンプル

### 5.1 `curl`（当日分 / デフォルト動作）

```bash
curl -s 'https://huggingface.co/api/daily_papers'
```

`date` を指定しなくても配列が返る例が確認できます。citeturn1search5  

### 5.2 `curl`（日付 + ページング）

```bash
curl -s 'https://huggingface.co/api/daily_papers?date=2025-11-10&page=1&limit=1'
```

この形式の例は非公式ドキュメントに記載があります。citeturn1search7  

---

## 6. 注意点（運用上のメモ）

- 本 API は公式な詳細スキーマが明文化されていない場合があるため、**実レスポンスを前提にフィールドが増減する**可能性があります（フロント実装都合で変わる）。citeturn1search5turn1search7  
- Hugging Face Hub の API はレート制限対象です。大量取得や定期ポーリングをする場合は、バックオフやキャッシュを入れるのが安全です。citeturn1search18  

---

## 7. 参考リンク

- `GET /api/daily_papers`（実レスポンスが見える）: https://huggingface.co/api/daily_papers citeturn1search5  
- （非公式）Hugging Face Papers API Documentation: https://github.com/0x0is1/hf-papers-api-docs citeturn1search7  
- Hub API Endpoints / OpenAPI 仕様への導線: https://huggingface.co/docs/hub/en/api citeturn1search18  
