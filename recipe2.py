# command line start: cat ./data.txt | prodigy space.ner.manual annotated_test2 - --loader txt --label PER,ORG,LOC -F ./recipe2.py
# command line export data: prodigy db-out annotated_test2 > ./annotations.jsonl

"""This recipe requires Prodigy v1.10+."""
from typing import List, Optional, Union, Iterable, Dict, Any
from prodigy.components.loaders import get_stream
from prodigy.util import get_labels
import prodigy


@prodigy.recipe(
    "space.ner.manual",    
    # fmt: off
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    tokenizer_vocab=("Tokenizer vocab file", "option", "tv", str),
    lowercase=("Set lowercase=True for tokenizer", "flag", "LC", bool),
    #hide_special=("Hide SEP and CLS tokens visually", "flag", "HS", bool),
    #hide_wp_prefix=("Hide wordpieces prefix like ##", "flag", "HW", bool)
    # fmt: on
)
def ner_manual_tokenizers_space(     
    dataset: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    tokenizer_vocab: Optional[str] = None,
    lowercase: bool = False,
    #hide_special: bool = False,
    #hide_wp_prefix: bool = False,
) -> Dict[str, Any]:
    """Example recipe that shows how to use model-specific tokenizers like the
    BERT word piece tokenizer to preprocess your incoming text for fast and
    efficient NER annotation and to make sure that all annotations you collect
    always map to tokens and can be used to train and fine-tune your model
    (even if the tokenization isn't that intuitive, because word pieces). The
    selection automatically snaps to the token boundaries and you can double-click
    single tokens to select them.
    Setting "honor_token_whitespace": true will ensure that whitespace between
    tokens is only shown if whitespace is present in the original text. This
    keeps the text readable.
    Requires Prodigy v1.10+ and uses the HuggingFace tokenizers library."""
    stream = get_stream(source, loader=loader, input_key="text")
    # You can replace this with other tokenizers if needed
    
    from nltk.tokenize import word_tokenize
    


    def add_tokens(stream):
        for eg in stream:
            s=eg["text"]            
            from nltk.tokenize import WhitespaceTokenizer
            # Create a reference variable for Class WhitespaceTokenizer
            tk = WhitespaceTokenizer()
            tokens= tk.tokenize(s)
            offsets= list(WhitespaceTokenizer().span_tokenize(s)) 
            eg_tokens = []
            idx = 0
            for (text, (start, end)) in zip(
                tokens, offsets
            ):
                # If we don't want to see special tokens, don't add them
                #if hide_special and text in special_tokens:
                #    continue
                # If we want to strip out word piece prefix, remove it from text
                # if hide_wp_prefix and wp_prefix is not None:
                #     if text.startswith(wp_prefix):
                #         text = text[len(wp_prefix):]
                token = {
                    "text": text,
                    "id": idx,
                    "start": start,
                    "end": end,
                    # Don't allow selecting spacial SEP/CLS tokens
                    "disabled": False #text in special_tokens,
                }
                eg_tokens.append(token)
                idx += 1
            for i, token in enumerate(eg_tokens):
                # If the next start offset != the current end offset, we
                # assume there's whitespace in between
                if i < len(eg_tokens) - 1: #and token["text"] not in special_tokens:
                    next_token = eg_tokens[i + 1]
                    token["ws"] = (
                        next_token["start"] > token["end"]
                        #or next_token["text"] in special_tokens
                    )
                else:
                    token["ws"] = True
            eg["tokens"] = eg_tokens
            yield eg

    stream = add_tokens(stream)

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "ner_manual",
        "config": {
            "honor_token_whitespace": True,
            "labels": label,
            "exclude_by": "input",
            "force_stream_order": True,
        },
    }