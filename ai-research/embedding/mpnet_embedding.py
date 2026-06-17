from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer,util

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


MODEL_BASE_PATH = "../model-base/all-mpnet-base-v2"


def transformer_embedding(sentences):
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_PATH)
    model = AutoModel.from_pretrained(MODEL_BASE_PATH,trust_remote_code=False)

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    print(f'embedding verctor sharp size:{sentence_embeddings.shape}')
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print(f"Sentence embeddings-{sentence_embeddings.shape}:{sentence_embeddings}")

def sentence_transformers_embedding(sentences):

    embeddings = model.encode(sentences,convert_to_tensor=True)
    print(f"sentence_transformers embeddings-{embeddings.shape}:{embeddings}")
    return embeddings

if __name__ == '__main__':
    model = SentenceTransformer(MODEL_BASE_PATH)
    # Sentences we want sentence embeddings for
    sentences = ["I love Python programming","I love Python programming","I love Python","I dont like Python",
    "Python is my favorite language",
    "I enjoy coding in Java",
    "I hate python programming",
    "nobody like python",
    "xiaomi like python",
    "i dislike python",
    "Cats are cute animals"]
    # transformer_embedding(sentences)
    # print("*"*50)
    corpus_embeddings = sentence_transformers_embedding(sentences)

    query_sentence = "I like Python"
    query_sentence_embeddings = sentence_transformers_embedding(query_sentence)

    similarities = util.pytorch_cos_sim(query_sentence_embeddings, corpus_embeddings)[0]
    top_results = torch.topk(similarities, k=5)

    for idx, score in zip(top_results.indices, top_results.values):
        print(f"  {score:.3f} - {sentences[idx]}")

    top_results1 = util.semantic_search(query_sentence_embeddings, corpus_embeddings, top_k=5)
    for result in top_results1[-1]:
        print(f'{result['corpus_id']}:{result['score']:.3f} -sentence:{sentences[result['corpus_id']]}')

    pairs = util.paraphrase_mining_embeddings(corpus_embeddings, top_k=3)
    print(pairs)
    for score, idx1, idx2 in pairs:
        print(f"相似度: {score:.4f}  [{idx1}] - {sentences[idx1]} & [{idx2}] {sentences[idx2]}")
