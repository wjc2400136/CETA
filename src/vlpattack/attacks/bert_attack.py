import copy
import itertools

import torch
import torch.nn as nn

filter_words = [
    "a",
    "about",
    "above",
    "across",
    "after",
    "afterwards",
    "again",
    "against",
    "ain",
    "all",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "am",
    "among",
    "amongst",
    "an",
    "and",
    "another",
    "any",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anywhere",
    "are",
    "aren",
    "aren't",
    "around",
    "as",
    "at",
    "back",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "couldn",
    "couldn't",
    "d",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "don",
    "don't",
    "down",
    "due",
    "during",
    "either",
    "else",
    "elsewhere",
    "empty",
    "enough",
    "even",
    "ever",
    "everyone",
    "everything",
    "everywhere",
    "except",
    "first",
    "for",
    "former",
    "formerly",
    "from",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "he",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "however",
    "hundred",
    "i",
    "if",
    "in",
    "indeed",
    "into",
    "is",
    "isn",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "latter",
    "latterly",
    "least",
    "ll",
    "may",
    "me",
    "meanwhile",
    "mightn",
    "mightn't",
    "mine",
    "more",
    "moreover",
    "most",
    "mostly",
    "must",
    "mustn",
    "mustn't",
    "my",
    "myself",
    "namely",
    "needn",
    "needn't",
    "neither",
    "never",
    "nevertheless",
    "next",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "now",
    "nowhere",
    "o",
    "of",
    "off",
    "on",
    "once",
    "one",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "per",
    "please",
    "s",
    "same",
    "shan",
    "shan't",
    "she",
    "she's",
    "should've",
    "shouldn",
    "shouldn't",
    "somehow",
    "something",
    "sometime",
    "somewhere",
    "such",
    "t",
    "than",
    "that",
    "that'll",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "thereupon",
    "these",
    "they",
    "this",
    "those",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "too",
    "toward",
    "towards",
    "under",
    "unless",
    "until",
    "up",
    "upon",
    "used",
    "ve",
    "was",
    "wasn",
    "wasn't",
    "we",
    "were",
    "weren",
    "weren't",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "with",
    "within",
    "without",
    "won",
    "won't",
    "would",
    "wouldn",
    "wouldn't",
    "y",
    "yet",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "a the",
]
filter_words = set(filter_words)

# The VLP author releases apply these entries as exact-token filters.  Only
# WordPiece continuation markers are substring-filtered below.
filter_words.update(
    {
        ".",
        "-",
        "/",
        "?",
        "some",
        '"',
        ",",
        "b",
        "&",
        "!",
        "@",
        "%",
        "^",
        "*",
        "(",
        ")",
        "+",
        "=",
        "<",
        ">",
        "|",
        ":",
        ";",
        "～",
        "·",
    }
)
filter_characters = {
    "##",
}


def tokenize_with_offset(tokenizer, text, offset):
    """Tokenize release-style space-separated words with an explicit MLM offset."""
    words = text.split(" ")
    sub_words = []
    keys = []
    index = offset
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
    return words, sub_words, keys


class BertAttack:
    def __init__(
        self,
        model_name,
        model,
        ref_model,
        tokenizer,
        max_length=30,
        threshold=0.3,
        num_perturb=1,
        topk=10,
        batch_size=32,
    ):
        self.model_name = model_name
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.threshold = threshold
        self.num_perturb = num_perturb
        self.topk = topk
        self.batch_size = batch_size
        self.criterion = torch.nn.KLDivLoss(reduction="none")
        self.device = next(self.model.parameters()).device

    def attack(self, images, texts, *args):
        texts_input = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        # (batch_size, max_length, vocab_size)
        mlm_logits = self.ref_model(
            texts_input.input_ids, attention_mask=texts_input.attention_mask
        ).logits
        # (batch_size, max_length, topk)
        word_pred_scores_all, word_predictions = torch.topk(mlm_logits, self.topk, -1)

        orig_feat = self.get_text_feat(texts)

        adv_texts = []
        for i, text in enumerate(texts):
            important_scores = self.get_important_scores(text, orig_feat[i])
            list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
            words, _, keys = self._tokenize(text)
            final_words = copy.deepcopy(words)

            change = 0
            for top_index, _ in list_of_index:
                if change >= self.num_perturb:
                    break

                target_word = words[top_index]
                if target_word in filter_words:
                    continue
                if keys[top_index][0] > self.max_length - 2:
                    continue

                substitutes = word_predictions[i, keys[top_index][0] : keys[top_index][1]]
                word_pred_scores = word_pred_scores_all[i, keys[top_index][0] : keys[top_index][1]]

                substitutes = get_substitues(
                    substitutes,
                    self.tokenizer,
                    self.ref_model,
                    True,
                    word_pred_scores,
                    self.threshold,
                )

                replace_texts = [" ".join(final_words)]
                available_substitutes = [target_word]
                for substitute in substitutes:
                    if substitute == target_word:
                        continue
                    if any(character in substitute for character in filter_characters):
                        continue
                    if substitute in filter_words:
                        continue

                    temp_replace = copy.deepcopy(final_words)
                    temp_replace[top_index] = substitute
                    available_substitutes.append(substitute)
                    replace_texts.append(" ".join(temp_replace))

                replace_feat = self.get_text_feat(replace_texts)
                loss = self.get_loss(replace_feat, orig_feat[i])
                candidate_idx = loss.argmax()

                final_words[top_index] = available_substitutes[candidate_idx]

                if available_substitutes[candidate_idx] != target_word:
                    change += 1

            adv_texts.append(" ".join(final_words))

        return images, adv_texts

    def get_text_feat(self, texts):
        with torch.no_grad():
            text_input = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            text_output = self.model.inference_text(text_input)
            text_feat = text_output["text_feat"]
        return text_feat

    def get_loss(self, replace_feat, orig_feat):
        loss = self.criterion(
            replace_feat.log_softmax(dim=-1),
            orig_feat.softmax(dim=-1).expand(len(replace_feat), -1),
        )
        loss = loss.sum(dim=-1)
        return loss

    def _tokenize(self, text):
        # The original BERT-Attack release slices away the leading [CLS]
        # prediction before indexing words. Indexing the unsliced MLM tensor
        # from one is the exact equivalent used by this adapter.
        return tokenize_with_offset(self.tokenizer, text, 1)

    def _get_masked(self, text):
        words = text.split(" ")
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ["[UNK]"] + words[i + 1 :])
        return masked_words

    def get_important_scores(self, text, orig_feat):
        masked_words = self._get_masked(text)
        masked_texts = [" ".join(words) for words in masked_words]

        masked_feats = []
        for i in range(0, len(masked_texts), self.batch_size):
            masked_feat = self.get_text_feat(masked_texts[i : i + self.batch_size])
            masked_feats.append(masked_feat)
        masked_feats = torch.cat(masked_feats, dim=0)
        important_scores = self.get_loss(masked_feats, orig_feat)
        return important_scores

    def __call__(self, *input, **kwargs):
        self.model.eval()
        return self.attack(*input, **kwargs)


def get_substitues(
    substitutes,
    tokenizer,
    mlm_model,
    use_bpe,
    substitutes_score,
    threshold,
    metadata_sink=None,
):
    words = []
    sub_len, _ = substitutes.size()

    if sub_len == 0:
        return words
    elif sub_len == 1:
        for i, j in zip(substitutes[0], substitutes_score[0], strict=False):
            if threshold != 0 and j < threshold:
                break
            token_id = int(i)
            token = tokenizer._convert_id_to_token(token_id)
            words.append(token)
            if metadata_sink is not None:
                metadata_sink.append(
                    {
                        "kind": "single_wordpiece_prediction",
                        "token_ids": [token_id],
                        "wordpieces": [token],
                        "decoded_string": token,
                    }
                )
    else:
        if use_bpe:
            words = get_bpe_substitues(
                substitutes,
                tokenizer,
                mlm_model,
                metadata_sink=metadata_sink,
            )
        else:
            return words
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model, metadata_sink=None):
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]

    lists_of_candidates = [row.tolist() for row in substitutes]
    all_substitutes = [list(p) for p in itertools.product(*lists_of_candidates)]
    all_substitutes = torch.tensor(all_substitutes)
    all_substitutes = all_substitutes[:24].to(device)

    N, L = all_substitutes.size()
    loss = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        word_predictions = mlm_model(all_substitutes)[0]
        perplexity = loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
        perplexity = torch.exp(torch.mean(perplexity.view(N, L), dim=-1))

    _, word_list = torch.sort(perplexity)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
        if metadata_sink is not None:
            metadata_sink.append(
                {
                    "kind": "multi_wordpiece_perplexity_candidate",
                    "token_ids": [int(value) for value in word.tolist()],
                    "wordpieces": tokens,
                    "decoded_string": text,
                }
            )
    return final_words
