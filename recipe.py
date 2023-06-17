# command line start: cat ./data.txt | prodigy space.ner.manual annotated_space - --loader txt --label PER,ORG,LOC -F ./recipe.py
# command line export data: prodigy db-out annotated_space > ./annotations.jsonl
# used for FPA 



"""This recipe requires Prodigy v1.10+."""
from typing import List, Optional, Union, Iterable, Dict, Any
from prodigy.components.loaders import get_stream
from prodigy.util import get_labels
import prodigy

from nltk.tokenize import WhitespaceTokenizer
import re


@prodigy.recipe(
    "space.ner.manual",    
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
    tokenizer_vocab=("Tokenizer vocab file", "option", "tv", str),
    lowercase=("Set lowercase=True for tokenizer", "flag", "LC", bool),

)
def ner_manual_tokenizers_space(     
    dataset: str,
    source: Union[str, Iterable[dict]],
    loader: Optional[str] = None,
    label: Optional[List[str]] = None,
    tokenizer_vocab: Optional[str] = None,
    lowercase: bool = False,

) -> Dict[str, Any]:

    stream = get_stream(source, loader=loader, input_key="text")
      


    def add_tokens(stream):
        for eg in stream:
            s=eg["text"]        
            
            s = re.sub("([.,,’‘“”/!?()--'])", r' \1 ', s) # seperate punctuation characters of interest with whitespaces
            s = s.replace('"', r' " ') # seperate punctuation characters of interest with whitespaces
            s = s.replace("'", r" ' ") # seperate punctuation characters of interest with whitespaces
            s = re.sub('\s{2,}', ' ', s)
            
            # Create a reference variable for Class WhitespaceTokenizer
            tk = WhitespaceTokenizer()
            tokens= tk.tokenize(s)
            offsets= list(WhitespaceTokenizer().span_tokenize(s)) 
            eg_tokens = []
            idx = 0
            for (text, (start, end)) in zip(
                tokens, offsets
            ):
                
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