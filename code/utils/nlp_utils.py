import re

def _tokenize(text, token_pattern=" "):
    if token_pattern == " ":
        return text.split(" ")
    else:
        token_pattern = re.compile(token_pattern, flags=re.UNICODE|re.LOCALE)
        return token_pattern.findall(text)
