from locale import normalize
from .utils import LIBRARY, preprocess, ADDITIONAL_TOKENS

def add(tokenizer):
    tokenizer.add_tokens(
        [f"<{key}>" for key in list(LIBRARY.keys()) + ADDITIONAL_TOKENS],
    )
    
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                "<MOJITO>",
                "</MOJITO>",
            ]
        }
    )
        
        

    