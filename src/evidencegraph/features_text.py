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


def init_language(language):
    assert language in ['en', 'de']
    if language == 'de':
        nlp = spacy.load(language)
        connectives = connectives_de
    elif language == 'en':
        nlp = spacy.load(language)
        connectives = connectives_en
    return TextFeatures(nlp=nlp, connectives=connectives)


def generate_items_segments(segments):
    """
    Generates a segment-wise item index for a list of segments.

    This method is used both by the featurizer as well as by the BaseClassifiers
    to generate a common order of input items.

    >>> segments = ["One,", "but two.", "Though three", "and four."]
    >>> generate_items_segments(segments)
    [1, 2, 3, 4]
    """
    return range(len(segments) + 1)[1:]


def generate_items_segmentpairs(segments):
    """
    Generates a segment-pair-wise item index for a list of segments.

    This method is used both by the featurizer as well as by the BaseClassifiers
    to generate a common order of input items.

    >>> segments = ["One,", "but two.", "Though three", "and four."]
    >>> generate_items_segmentpairs(segments)
    [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)]
    """
    return sorted(list(permutations(generate_items_segments(segments), 2)))



class TextFeatures(object):

    F_SET_BOW_BASELINE = ['default', 'bow']
    F_SET_ALL_BUT_VECTORS = [
        'default', 'bow', 'bow_2gram', 'first_three',
        'tags', 'deps_lemma', 'deps_tag',
        'punct', 'verb_main', 'verb_all', 'discourse_marker',
        'context', 'clusters', 'clusters_2gram', 'discourse_relation',
        'vector_left_right', 'vector_source_target',
        'verb_segment', 'same_sentence', 'matrix_clause'
    ]

    def __init__(self, nlp=None, connectives=None, feature_set=None):
        """
        A featurizer producing dict-features for text input.

        :param nlp: a spacy nlp for the desired language.
        :param connectives: a dictionary of discource connectives of the
            desired language mapping to potentially signalled discourse
            relations.
        :param feature_set: a list of strings defining the features to be used
        """
        self.nlp = nlp
        self.connectives = connectives or {}
        # caching
        self.doc_cache = dict()
        self.seg_cache = defaultdict(dict)
        self.idx_cache = defaultdict(dict)
        self.feature_set = feature_set or ["default"]
        self.feature_set_allowed_for_context = [
            'default', 'bow', 'first_three', 'punct',
            'verb_main', 'verb_segment', 'verb_all',
            'discourse_marker', 'discourse_relation'
        ]

    def feature_function_segments(self, segments, **kwargs):
        """
        A feature function returning segment-wise features for every of the given segments.

        >>> features = init_language('en')
        >>> segments = ['Hi there!', 'My name is Peter.']
        >>> f = features.feature_function_segments(segments, feature_set=['clusters_2gram'])
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
        self.parse(segments)
        for segment in generate_items_segments(segments):
            l.append(self.feature_function_single_segment(segments, segment, **kwargs))
        return l


    def feature_function_segmentpairs(self, segments, **kwargs):
        """
        A feature function retuning segment-pair-wise features for every possible pair of
        the given segments.

        >>> features = init_language('en')
        >>> segments = ['Hi there!', 'My name is Peter.']
        >>> f = features.feature_function_segmentpairs(segments, feature_set=['bow'])
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
        self.parse(segments)
        for source, target in generate_items_segmentpairs(segments):
            f = self.feature_function_single_segmentpair(segments, source, target, **kwargs)
            f = add_prefixed_dict(f, 'source', self.feature_function_single_segment(segments, source, **kwargs))
            f = add_prefixed_dict(f, 'target', self.feature_function_single_segment(segments, target, **kwargs))
            l.append(f)
        return l

    def feature_function_single_segment(self, segments, segment, feature_set=None):
        """
        Returns the segment-wise features for a single segment.

        >>> features = init_language('en')
        >>> segments = [u'Hi there! ', u'My name is Peter. ', u'Therefore I am happy.']
        >>> features.parse(segments)

        >>> f = features.feature_function_single_segment(segments, 1, feature_set=['default'])
        >>> sorted(f.items())
        [('POS_abs', 1), ('POS_first', True), ('POS_last', False), ('POS_rel', 0.333...)]

        >>> f = features.feature_function_single_segment(segments, 1, feature_set=['bow'])
        >>> sorted(f.items())
        [(u'TOK_!', True), (u'TOK_hi', True), (u'TOK_there', True)]

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['first_three'])
        >>> sorted(f.keys())
        [u'F3L_1_my', u'F3L_2_name', u'F3L_3_be']

        >>> f = features.feature_function_single_segment(segments, 1, feature_set=['clusters'])
        >>> sorted(f.items())
        [('CLS_0', True), ('CLS_1726', True), ('CLS_986', True)]

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['tags'])
        >>> sorted(f.keys())
        ['TAG_.', 'TAG_NN', 'TAG_NNP', 'TAG_PRP$', 'TAG_VBZ']

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['deps_lemma'])
        >>> sorted(f.keys())
        [u'DPL_._punct_be', u'DPL_be_ROOT_be', u'DPL_my_poss_name', u'DPL_name_nsubj_be', u'DPL_peter_attr_be']

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['deps_tag'])
        >>> sorted(f.keys())
        ['DPT_._punct_VBZ', 'DPT_NNP_attr_VBZ', 'DPT_NN_nsubj_VBZ', 'DPT_PRP$_poss_NN', 'DPT_VBZ_ROOT_VBZ']

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['punct'])
        >>> sorted(f.items())
        [('punctuation_count', 1)]

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['verb_main'])
        >>> sorted(f.items())
        [(u'VM_lemma_be', True), (u'VM_text_is', True)]

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['verb_all'])
        >>> sorted(f.items())
        [(u'VA_lemma_be', True), (u'VA_text_is', True)]

        >>> f = features.feature_function_single_segment(segments, 3, feature_set=['discourse_marker', 'discourse_relation'])
        >>> sorted(f.items())
        [(u'DM_therefore', True), (u'DR_result', True)]

        >>> f = features.feature_function_single_segment(segments, 2, feature_set=['vector_left_right'])
        >>> sorted(f.items())
        [('VR_left', 0.401...), ('VR_right', 0.328...)]
        """
        if feature_set is None:
            feature_set = self.feature_set
        d = {}
        tokens = self.get_tokens(segments, segment)

        if 'default' in feature_set:
            d['POS_abs'] = segment
            d['POS_rel'] = float(segment) / len(segments)
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
            for marker, relations in self.connectives.iteritems():
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
                d['VR_left'] = self.vector_similarity(segments, segment, segment - 1)
            if segment < len(segments):
                d['VR_right'] = self.vector_similarity(segments, segment, segment + 1)

        if 'context' in feature_set:
            new_feature_set = [
                f for f in feature_set if f in self.feature_set_allowed_for_context]
            if segment > 1:
                d = add_prefixed_dict(d, 'left', self.feature_function_single_segment(segments, segment-1, feature_set=new_feature_set))
            if segment < len(segments):
                d = add_prefixed_dict(d, 'right', self.feature_function_single_segment(segments, segment+1, feature_set=new_feature_set))

        return d

    def feature_function_single_segmentpair(self, segments, source, target, feature_set=None):
        """
        Returns the segment-pair-wise features for a single pair of segments.

        >>> features = init_language('en')
        >>> segments = [u'Hi there! ', u'My name is Peter.']
        >>> features.parse(segments)
        >>> f = features.feature_function_single_segmentpair(segments, 2, 1, feature_set=['vector_source_target'])
        >>> sorted(f.items())
        [('VR_src_trg', 0.401...), ('direction', False), ('distance', -1),
        ('distance_abs', 1), ('distance_rel', 0.5), ('segment_length_ratio', 1.666...)]
        """
        if feature_set is None:
            feature_set = self.feature_set
        d = {}
        distance = target - source
        d['distance'] = distance
        d['distance_abs'] = abs(distance)
        d['distance_rel'] = 1.0 * abs(distance) / len(segments)
        d['direction'] = distance > 0
        d['segment_length_ratio'] = 1.0 * len(self.get_tokens(segments, source)) / len(self.get_tokens(segments, target))

        if 'vector_source_target' in feature_set:
            d['VR_src_trg'] = self.vector_similarity(segments, source, target)

        if 'same_sentence' in feature_set:
            d['same_sentence'] = self.same_sentence(segments, source, target)

        if 'matrix_clause' in feature_set:
            d['matrix_clause'] = self.matrix_clause(segments, source, target)

        return d

    def get_tokens(self, segments, segment):
        text = text_of_segments(segments)
        return self.seg_cache[text][segment]

    def preparse(self, segments):
        self.parse(add_segment_final_space(ensure_unicode(segments)))

    def parse(self, segments):
        """
        >>> features = init_language('en')
        >>> segments = [u'Hi there! ', u'My name is Peter.']
        >>> features.parse(segments)
        >>> text = u'Hi there! My name is Peter.'
        >>> text in features.doc_cache
        True
        >>> [s in features.seg_cache[text] for s in [1,2]]
        [True, True]
        """
        text = text_of_segments(segments)
        if text not in self.doc_cache:
            # parse whole text
            doc = list(self.nlp.pipe([text]))[0]
            self.doc_cache[text] = doc
            # match segments with parsed tokens
            segments_to_match = deque(segments[:])
            iterator = iter(doc)
            count_segments = 1
            consumed_tokens = u''
            tokens = []
            while segments_to_match:
                if consumed_tokens == segments_to_match[0]:
                    segments_to_match.popleft()
                    self.seg_cache[text][count_segments] = tokens
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
                    self.idx_cache[text][token.idx] = sentence_id

    def same_sentence(self, segments, source, target):
        text = text_of_segments(segments)
        src_tokens = self.get_tokens(segments, source)
        trg_tokens = self.get_tokens(segments, target)
        sentence_id_of_token = self.idx_cache[text]
        sentence_ids_of_source_tokens = set(sentence_id_of_token[t.idx] for t in src_tokens)
        sentence_ids_of_target_tokens = set(sentence_id_of_token[t.idx] for t in trg_tokens)
        return sentence_ids_of_source_tokens == sentence_ids_of_target_tokens

    def matrix_clause(self, segments, source, target):
        src_tokens = self.get_tokens(segments, source)
        trg_tokens = self.get_tokens(segments, target)
        return any(token.head in trg_tokens for token in src_tokens if not token.is_punct)

    def vector_similarity(self, segments, seg1, seg2):
        vec1 = average_vector_of_segment(self.get_tokens(segments, seg1))
        vec2 = average_vector_of_segment(self.get_tokens(segments, seg2))
        return cosine(vec1, vec2)


### HELPER METHODS


def text_of_segments(segments):
    """
    >>> segments = [u'Hi there! ', u'My name is Peter.']
    >>> text_of_segments(segments)
    u'Hi there! My name is Peter.'
    """
    return u''.join(segments)



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


def average_vector_of_segment(tokens):
    return average([
        token.vector for token in tokens
        if token.has_vector and not (token.is_stop or token.is_space)])
