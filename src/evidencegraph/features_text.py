# -*- mode: python; coding: utf-8; -*-

'''
Created on 20.05.2016

@author: Andreas Peldszus
'''


from itertools import permutations
from collections import defaultdict
from collections import deque

from scipy.spatial.distance import cosine
import spacy

from utils import window
from resources import connectives_en
from resources import connectives_de


nlp = None

doc_cache = dict()
seg_cache = defaultdict(dict)
feature_cache = dict()
idx_cache = defaultdict(dict)


def init_language(language):
    global nlp, connectives
    assert language in ['en', 'de']
    if language == 'de':
        nlp = spacy.load(language)
        connectives = connectives_de
    elif language == 'en':
        nlp = spacy.load(language)
        connectives = connectives_en


def get_tokens(segments, segment):
    text = text_of_segments(segments)
    return seg_cache[text][segment]


def text_of_segments(segments):
    """
    >>> segments = [u'Hi there! ', u'My name is Peter.']
    >>> text_of_segments(segments)
    u'Hi there! My name is Peter.'
    """
    return u''.join(segments)


def preparse(segments):
    parse(add_segment_final_space(ensure_unicode(segments)))


def parse(segments):
    """
    >>> segments = [u'Hi there! ', u'My name is Peter.']
    >>> parse(segments)
    >>> text = u'Hi there! My name is Peter.'
    >>> text in doc_cache
    True
    >>> [s in seg_cache[text] for s in [1,2]]
    [True, True]
    """
    text = text_of_segments(segments)
    if text not in doc_cache:
        # parse whole text
        # print '.'
        if nlp is None:
            init_language('en')
        doc = list(nlp.pipe([text]))[0]
        doc_cache[text] = doc
        # match segments with parsed tokens
        segments_to_match = deque(segments[:])
        iterator = iter(doc)
        count_segments = 1
        consumed_tokens = u''
        tokens = []
        while segments_to_match:
            if consumed_tokens == segments_to_match[0]:
                segments_to_match.popleft()
                seg_cache[text][count_segments] = tokens
                count_segments += 1
                consumed_tokens = u''
                tokens = []
            else:
                try:
                    token = next(iterator)
                except Exception:
                    print segments
                tokens.append(token)
                consumed_tokens += token.string
        # build offset -> sentence mapping
        for sentence_id, sentence in enumerate(doc.sents):
            for token in sentence:
                idx_cache[text][token.idx] = sentence_id


def ensure_unicode(segments):
    """
    >>> segments = ['Hi there!', 'My name is Peter.']
    >>> ensure_unicode(segments)
    [u'Hi there!', u'My name is Peter.']
    """
    return [unicode(s) for s in segments]


def add_segment_final_space(segments):
    """
    >>> segments = ['Hi there!', 'Here... ', 'My name is Peter. ']
    >>> add_segment_final_space(segments)
    ['Hi there! ', 'Here... ', 'My name is Peter.']
    """
    r = []
    for segment in segments[:-1]:
        r.append(segment.rstrip()+' ')
    r.append(segments[-1].rstrip())
    return r


def average(vectors):
    assert len(vectors) > 0
    r = sum(vectors)
    r /= len(vectors)
    return r


def add_prefixed_dict(output_dictionary, prefix, input_dictionary, copy=False):
    if copy:
        output_dictionary = copy(output_dictionary)
    for key, value in input_dictionary.iteritems():
        assert isinstance(key, basestring)
        output_dictionary[prefix+'_'+key] = value
    return output_dictionary


def bucket(number):
    if 0 <= number <= 2:
        return number
    else:
        return 3


def bucket_percent(number):
    if number > 0.8:
        return 5
    elif number > 0.6:
        return 4
    elif number > 0.4:
        return 3
    elif number > 0.2:
        return 2
    else:
        return 1

feature_set_allowed_for_context = [
        'default', 'bow', 'first_three', 'punct',
        'verb_main', 'verb_segment', 'verb_all',
        'discourse_marker', 'discourse_relation']


def average_vector_of_segment(tokens):
    return average([
        token.vector for token in tokens
        if token.has_vector and not (token.is_stop or token.is_space)])


def vector_similarity(segments, seg1, seg2):
    vec1 = average_vector_of_segment(get_tokens(segments, seg1))
    vec2 = average_vector_of_segment(get_tokens(segments, seg2))
    return cosine(vec1, vec2)


def feature_function_for_segments(segments, segment, feature_set=['default']):
    """
    >>> segments = [u'Hi there! ', u'My name is Peter. ', u'Therefore I am happy.']
    >>> parse(segments)

    >>> f = feature_function_for_segments(segments, 1, feature_set=['default'])
    >>> sorted(f.items())
    [('POS_abs', 1), ('POS_first', True), ('POS_last', False), ('POS_rel', 0.333...)]

    >>> f = feature_function_for_segments(segments, 1, feature_set=['bow'])
    >>> sorted(f.items())
    [(u'TOK_!', True), (u'TOK_hi', True), (u'TOK_there', True)]

    >>> f = feature_function_for_segments(segments, 2, feature_set=['first_three'])
    >>> sorted(f.keys())
    [u'F3L_1_my', u'F3L_2_name', u'F3L_3_be']

    >>> f = feature_function_for_segments(segments, 1, feature_set=['clusters'])
    >>> sorted(f.items())
    [('CLS_0', True), ('CLS_1726', True), ('CLS_986', True)]

    >>> f = feature_function_for_segments(segments, 2, feature_set=['tags'])
    >>> sorted(f.keys())
    ['TAG_.', 'TAG_NN', 'TAG_NNP', 'TAG_PRP$', 'TAG_VBZ']

    >>> f = feature_function_for_segments(segments, 2, feature_set=['deps_lemma'])
    >>> sorted(f.keys())
    [u'DPL_._punct_be', u'DPL_be_ROOT_be', u'DPL_my_poss_name', u'DPL_name_nsubj_be', u'DPL_peter_attr_be']

    >>> f = feature_function_for_segments(segments, 2, feature_set=['deps_tag'])
    >>> sorted(f.keys())
    ['DPT_._punct_VBZ', 'DPT_NNP_attr_VBZ', 'DPT_NN_nsubj_VBZ', 'DPT_PRP$_poss_NN', 'DPT_VBZ_ROOT_VBZ']

    >>> f = feature_function_for_segments(segments, 2, feature_set=['punct'])
    >>> sorted(f.items())
    [('punctuation_count', 1)]

    >>> f = feature_function_for_segments(segments, 2, feature_set=['verb_main'])
    >>> sorted(f.items())
    [(u'VM_lemma_be', True), (u'VM_text_is', True)]

    >>> f = feature_function_for_segments(segments, 2, feature_set=['verb_all'])
    >>> sorted(f.items())
    [(u'VA_lemma_be', True), (u'VA_text_is', True)]

    >>> f = feature_function_for_segments(segments, 3, feature_set=['discourse_marker', 'discourse_relation'])
    >>> sorted(f.items())
    [(u'DM_therefore', True), (u'DR_result', True)]

    >>> f = feature_function_for_segments(segments, 2, feature_set=['vector_left_right'])
    >>> sorted(f.items())
    [('VR_left', 0.401...), ('VR_right', 0.328...)]
    """
    d = {}
    tokens = get_tokens(segments, segment)

    if 'default' in feature_set:
        d['POS_abs'] = segment
        d['POS_rel'] = float(segment) / len(segments)
        # bucket_percent(float(segment) / len(segments))
        d['POS_first'] = segment == 1
        d['POS_last'] = segment == len(segments)

    if 'bow' in feature_set:
        for token in tokens:
            d[u'TOK_{}'.format(token.lemma_)] = True

    if 'bow_2gram' in feature_set:
        for tok1, tok2 in window(tokens, n=2):
            d[u'TOK_2_{}_{}'.format(tok1.lemma_, tok2.lemma_)] = True

    if 'first_three' in feature_set:
        for i, token in enumerate(tokens[:3], 1):
            d[u'F3L_{}_{}'.format(i, token.lemma_)] = True

    if 'clusters' in feature_set:
        for token in tokens:
            d['CLS_{}'.format(token.cluster)] = True

    if 'clusters_2gram' in feature_set:
        for tok1, tok2 in window(tokens, n=2):
            d[u'CLS_2_{}_{}'.format(tok1.cluster, tok2.cluster)] = True

    if 'vectors' in feature_set:
        for i, v in enumerate(average_vector_of_segment(tokens)):
            d['VEC_{}'.format(i)] = v

    if 'tags' in feature_set:
        for token in tokens:
            d['TAG_{}'.format(token.tag_)] = True

    if 'deps_lemma' in feature_set:
        for token in tokens:
            d[u'DPL_{}_{}_{}'.format(token.lemma_, token.dep_, token.head.lemma_)] = True

    if 'deps_tag' in feature_set:
        for token in tokens:
            d['DPT_{}_{}_{}'.format(token.tag_, token.dep_, token.head.tag_)] = True

    if 'punct' in feature_set:
        d['punctuation_count'] = sum(1 for token in tokens if token.is_punct)

    if 'verb_main' in feature_set:
        for token in tokens:
            if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
                d[u'VM_text_{}'.format(token.text)] = True
                d[u'VM_lemma_{}'.format(token.lemma_)] = True

    if 'verb_segment' in feature_set:
        for token in tokens:
            if token.pos_ == 'VERB' and token.head not in tokens:
                d[u'VS_text_{}'.format(token.text)] = True
                d[u'VS_lemma_{}'.format(token.lemma_)] = True

    if 'verb_all' in feature_set:
        for token in tokens:
            if token.pos_ == 'VERB':
                d[u'VA_text_{}'.format(token.text)] = True
                d[u'VA_lemma_{}'.format(token.lemma_)] = True

    if 'discourse_marker' in feature_set:
        seg_text = u''.join(token.string for token in tokens).lower()
        tok_texts = [token.text.lower() for token in tokens]
        for marker, relations in connectives_en.iteritems():
            match = False
            if u' ' in marker:
                if marker in seg_text:
                    match = True
            else:
                if marker in tok_texts:
                    match = True
            if match:
                d[u'DM_{}'.format(marker)] = True
                if 'discourse_relation' in feature_set:
                    for rel in relations:
                        d[u'DR_{}'.format(rel)] = True

    if 'vector_left_right' in feature_set:
        if segment > 1:
            d['VR_left'] = vector_similarity(segments, segment, segment - 1)
            # bucket_percent(vector_similarity(segments, segment, segment - 1))
        if segment < len(segments):
            d['VR_right'] = vector_similarity(segments, segment, segment + 1)
            # bucket_percent(vector_similarity(segments, segment, segment + 1))

    if 'context' in feature_set:
        new_feature_set = [
            f for f in feature_set if f in feature_set_allowed_for_context]
        if segment > 1:
            d = add_prefixed_dict(d, 'left', feature_function_for_segments(segments, segment-1, feature_set=new_feature_set))
        if segment < len(segments):
            d = add_prefixed_dict(d, 'right', feature_function_for_segments(segments, segment+1, feature_set=new_feature_set))

    return d


def same_sentence(segments, source, target):
    text = text_of_segments(segments)
    src_tokens = get_tokens(segments, source)
    trg_tokens = get_tokens(segments, target)
    sentence_id_of_token = idx_cache[text]
    sentence_ids_of_source_tokens = set(sentence_id_of_token[t.idx] for t in src_tokens)
    sentence_ids_of_target_tokens = set(sentence_id_of_token[t.idx] for t in trg_tokens)
    return sentence_ids_of_source_tokens == sentence_ids_of_target_tokens


def matrix_clause(segments, source, target):
    src_tokens = get_tokens(segments, source)
    trg_tokens = get_tokens(segments, target)
    return any(token.head in trg_tokens for token in src_tokens if not token.is_punct)


def feature_function_for_segmentpairs(segments, source, target, feature_set=['default']):
    """
    >>> segments = [u'Hi there! ', u'My name is Peter.']
    >>> parse(segments)
    >>> f = feature_function_for_segmentpairs(segments, 2, 1, feature_set=['vector_source_target'])
    >>> sorted(f.items())
    [('VR_src_trg', 0.401...), ('direction', False), ('distance', -1),
     ('distance_abs', 1), ('distance_rel', 0.5), ('segment_length_ratio', 1.666...)]
    """
    d = {}
    distance = target - source
    d['distance'] = distance
    d['distance_abs'] = abs(distance)
    d['distance_rel'] = 1.0 * abs(distance) / len(segments)
    # bucket_percent(1.0 * abs(distance) / len(segments))
    d['direction'] = distance > 0
    d['segment_length_ratio'] = 1.0 * len(get_tokens(segments, source)) / len(get_tokens(segments, target))
    # bucket_percent(1.0 * len(get_tokens(segments, source)) / len(get_tokens(segments, target)))

    if 'vector_source_target' in feature_set:
        d['VR_src_trg'] = vector_similarity(segments, source, target)
        # bucket_percent(vector_similarity(segments, source, target))

    if 'same_sentence' in feature_set:
        d['same_sentence'] = same_sentence(segments, source, target)

    if 'matrix_clause' in feature_set:
        d['matrix_clause'] = matrix_clause(segments, source, target)

    return d


def generate_items_segments(segments):
    return range(len(segments) + 1)[1:]


def generate_items_segmentpairs(segments):
    return sorted(list(permutations(generate_items_segments(segments), 2)))


def cached(func, *args, **kwargs):
    key = str((args, kwargs))
    if key in feature_cache:
        return feature_cache[key]
    else:
        result = func(*args, **kwargs)
        feature_cache[key] = result
        return result


def feature_function_segments(segments, **kwargs):
    """
    >>> segments = ['Hi there!', 'My name is Peter.']
    >>> f = feature_function_segments(segments, feature_set=['clusters_2gram'])
    >>> len(f) == 2
    True
    >>> sorted(f[0].items())
    [(u'CLS_2_1726_986', True), (u'CLS_2_986_0', True)]
    >>> sorted(f[1].items())
    [(u'CLS_2_4021_762', True), (u'CLS_2_502_8', True),
     (u'CLS_2_762_502', True), (u'CLS_2_94_4021', True)]
    """
    l = []
    segments = add_segment_final_space(ensure_unicode(segments))
    parse(segments)
    for segment in generate_items_segments(segments):
        l.append(feature_function_for_segments(segments, segment, **kwargs))
    return l


# def feature_function_segments(*args, **kwargs):
#     return cached(_feature_function_segments, *args, **kwargs)


def feature_function_segmentpairs(segments, **kwargs):
    """
    >>> segments = ['Hi there!', 'My name is Peter.']
    >>> f = feature_function_segmentpairs(segments, feature_set=['bow'])
    >>> len(f) == 2
    True
    >>> sorted(f[0].items())
    [('direction', True), ('distance', 1), ('distance_abs', 1),
     ('distance_rel', 0.5), ('segment_length_ratio', 0.6),
     (u'source_TOK_!', True), (u'source_TOK_hi', True),
     (u'source_TOK_there', True), (u'target_TOK_.', True),
     (u'target_TOK_be', True), (u'target_TOK_my', True),
     (u'target_TOK_name', True), (u'target_TOK_peter', True)]
    """
    l = []
    segments = add_segment_final_space(ensure_unicode(segments))
    parse(segments)
    for source, target in generate_items_segmentpairs(segments):
        f = feature_function_for_segmentpairs(segments, source, target, **kwargs)
        f = add_prefixed_dict(f, 'source', feature_function_for_segments(segments, source, **kwargs))
        f = add_prefixed_dict(f, 'target', feature_function_for_segments(segments, target, **kwargs))
        l.append(f)
    return l


# def feature_function_segmentpairs(*args, **kwargs):
#     return cached(_feature_function_segmentpairs, *args, **kwargs)
