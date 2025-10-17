class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels

    def __str__(self):
        return f"Guid=:{self.guid}, words:{self.words}, labels:{self.labels}"

def read_file(input_file):
    with open(input_file) as f:
        # out_lines = []
        out_lists = []
        entries = f.read().strip().split("\n\n")
        for entry in entries:
            words = []
            ner_labels = []
            pos_tags = []
            bio_pos_tags = []
            for line in entry.splitlines():
                pieces = line.strip().split()
                if len(pieces) < 1:
                    continue
                word = pieces[0]
                # if word == "-DOCSTART-" or word == '':
                #     continue
                words.append(word)
                pos_tags.append(pieces[1])
                bio_pos_tags.append(pieces[2])
                ner_labels.append(pieces[-1])
            # sentence = ' '.join(words)
            # ner_seq = ' '.join(ner_labels)
            # pos_tag_seq = ' '.join(pos_tags)
            # bio_pos_tag_seq = ' '.join(bio_pos_tags)
            # out_lines.append([sentence, pos_tag_seq, bio_pos_tag_seq, ner_seq])
            # out_lines.append([sentence, ner_seq])
            out_lists.append([words,pos_tags,bio_pos_tags,ner_labels])
    return out_lists

def _create_examples(all_lists):
    examples = []
    for (i, one_lists) in enumerate(all_lists):
        guid = i
        words = one_lists[0]
        labels = one_lists[-1]
        examples.append(InputExample(
            guid=guid, words=words, labels=labels))
    return examples


if __name__ == '__main__':
    INPUT_FILE ='./data/input/conll2003/train.txt'
    train_data = read_file(INPUT_FILE)
    # print(train_data[:10])
    for example in _create_examples(train_data[:10]):
        print(example)

    print(InputExample(guid="A", words="B", labels="C"))
