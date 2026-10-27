"""Deterministic positive-only unified sentence prompts and semantic spans."""
from dataclasses import dataclass

from .attributes import COLORS, attribute_vector

PROMPT_VERSION = 'upar-sentence-v1'


@dataclass(frozen=True)
class AttributeSpan:
    attribute_index: int
    start: int
    end: int


@dataclass(frozen=True)
class Prompt:
    attributes: tuple
    text: str
    spans: tuple


@dataclass(frozen=True)
class TokenizedPrompt:
    prompt: Prompt
    token_ids: tuple
    attribute_token_mask: tuple
    token_attribute_indices: tuple
    token_offsets: tuple
    length: int


class PromptTooLong(ValueError):
    pass


def _fixed(text):
    return (text, None)


def _value(text, index):
    return (text, index)


def _join(parts, separator=' and '):
    result = []
    for i, part in enumerate(parts):
        if i:
            result.append(_fixed(separator))
        result.extend(part)
    return result


def _colors(vector, offset):
    return _join([
        [_value(color.lower() if color != 'Other' else 'other-colored', offset + i)]
        for i, color in enumerate(COLORS) if vector[offset + i]
    ])


def _clothing(vector, upper):
    length_index, color_offset = (7, 8) if upper else (20, 21)
    values = []
    if vector[length_index]:
        values.append(_value('short', length_index))
    colors = _colors(vector, color_offset)
    if colors:
        if values:
            values.append(_fixed(' '))
        values.extend(colors)
    types = []
    if not upper:
        if vector[33]:
            types.append([_value('trousers', 33), _fixed(' or '), _value('shorts', 33)])
        if vector[34]:
            types.append([_value('skirts', 34), _fixed(' or '), _value('dresses', 34)])
    if not values and not types:
        return []
    if types:
        return values + ([_fixed(' ')] if values else []) + _join(types)
    noun = 'upper garment' if upper else 'lower clothing'
    if upper:
        first_color = next((i for i in range(8, 20) if vector[i]), None)
        article = 'an ' if not vector[7] and first_color in (13, 19) else 'a '
        return [_fixed(article)] + values + [_fixed(' ' + noun)]
    return values + [_fixed(' ' + noun)]


def generate_prompt(attributes):
    """The source dataset never enters this function."""
    vector = attribute_vector(attributes)
    clauses = []
    ages = [[_value(word, i)] for i, word in enumerate(('young', 'adult', 'old')) if vector[i]]
    if ages or vector[3]:
        if len(ages) == 1:
            age_index = ages[0][0][1]
            article = 'an ' if age_index == 1 else 'a '
            values = [_fixed('is ' + article)] + ages[0] + [_fixed(' ')]
            values += [_value('woman', 3)] if vector[3] else [_fixed('person')]
        elif not ages:
            values = [_fixed('is a '), _value('woman', 3)]
        else:
            values = [_fixed('is ')] + _join(ages)
            if vector[3]:
                values += [_fixed(' and '), _value('female', 3)]
        clauses.append(values)
    hair = [[_value(word, index)] for index, word in ((4, 'short hair'), (5, 'long hair')) if vector[index]]
    if hair:
        clauses.append([_fixed('has ')] + _join(hair))
    if vector[6]:
        clauses.append([_fixed('is '), _value('bald', 6)])
    clothing = [part for part in (_clothing(vector, True), _clothing(vector, False)) if part]
    if clothing:
        clauses.append([_fixed('wears ')] + _join(clothing))
    carrying = [[_fixed('a '), _value(word, index)]
                for index, word in ((35, 'backpack'), (36, 'bag')) if vector[index]]
    if carrying:
        clauses.append([_fixed('carries ')] + _join(carrying))
    wearing = []
    for index, word in ((37, 'regular glasses'), (38, 'sunglasses'), (39, 'hat')):
        if vector[index]:
            wearing.append(([_fixed('a ')] if index == 39 else []) + [_value(word, index)])
    if wearing:
        clauses.append([_fixed('wears ')] + _join(wearing))
    # Complete neutral sentence for an all-zero vector, without opposite attributes.
    if not clauses:
        return Prompt(vector, 'The pedestrian is visible.', ())
    segments = [_fixed('The pedestrian ')]
    for index, clause in enumerate(clauses):
        if index:
            segments.append(_fixed(' and ' if len(clauses) == 2 else
                                   ', and ' if index == len(clauses) - 1 else ', '))
        segments.extend(clause)
    segments.append(_fixed('.'))
    text, spans = '', []
    for surface, attribute_index in segments:
        start = len(text)
        text += surface
        if attribute_index is not None:
            spans.append(AttributeSpan(attribute_index, start, len(text)))
    return Prompt(vector, text, tuple(spans))


def tokenize_prompt(prompt, tokenizer, text_length=77):
    ids, offsets = tokenizer.encode_with_offsets(prompt.text)
    length = len(ids) + 2
    if length > text_length:
        raise PromptTooLong(
            f'Prompt needs {length} CLIP BPE tokens including SOS/EOS; limit is {text_length}. '
            f'Attributes cannot be silently truncated: {prompt.attributes}; text={prompt.text!r}')
    tags = []
    for start, end in offsets:
        owners = {span.attribute_index for span in prompt.spans
                  if start < span.end and end > span.start}
        if len(owners) > 1:
            raise ValueError('A BPE token overlaps distinct attribute expressions')
        if owners and not any(span.start <= start and end <= span.end
                              for span in prompt.spans if span.attribute_index in owners):
            raise ValueError('An attribute span splits a BPE token')
        # Punctuation inside lexical values (e.g. other-colored) is not masked.
        lexical = any(character.isalnum() for character in prompt.text[start:end])
        tags.append(next(iter(owners)) if owners and lexical else -1)
    padding = text_length - length
    token_ids = (tokenizer.sot_id, *ids, tokenizer.eot_id, *([0] * padding))
    tags = (-1, *tags, -1, *([-1] * padding))
    offsets = ((-1, -1), *offsets, (-1, -1), *([(-1, -1)] * padding))
    expected = {index for index, value in enumerate(prompt.attributes) if value}
    if {tag for tag in tags if tag >= 0} != expected:
        raise ValueError('Prompt token spans do not represent every positive attribute')
    return TokenizedPrompt(prompt, token_ids, tuple(tag >= 0 for tag in tags), tags, offsets, length)


def encode_query(attributes, tokenizer):
    return tokenize_prompt(generate_prompt(attributes), tokenizer)
