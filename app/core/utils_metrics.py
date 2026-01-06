import string
import re
import collections

# --- FUNGSI STANDAR (EM & F1) ---
def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_answer(prediction) == normalize_answer(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)

# --- FUNGSI : RAGAS METRICS (LLM-as-a-Judge) ---
class LocalRagasEvaluator:
    def __init__(self, generator_model):
        """
        Menggunakan model LLM yang sama (Mistral) untuk menjadi Juri.
        """
        self.llm = generator_model.llm # Instance llama-cpp

    def evaluate_faithfulness(self, answer, contexts):
        """
        Menilai Faithfulness: Apakah jawaban hanya berasal dari konteks? (0.0 - 1.0)
        """
        if not contexts: return 0.0
        
        context_text = "\n".join([c['text'] for c in contexts])[:1000]
        
        prompt = f"""[INST] You are an impartial judge. Evaluate if the ANSWER is derived ONLY from the CONTEXT provided.
        
        CONTEXT:
        {context_text}
        
        ANSWER:
        {answer}
        
        Is the answer faithful to the context?
        Reply ONLY with a score from 0.0 (Not Faithful) to 1.0 (Fully Faithful). Do not give explanation.
        Score: [/INST]"""
        
        return self._get_llm_score(prompt)

    def evaluate_relevancy(self, query, answer):
        """
        Menilai Answer Relevancy: Apakah jawaban nyambung dengan pertanyaan? (0.0 - 1.0)
        """
        prompt = f"""[INST] You are an impartial judge. Evaluate if the ANSWER is relevant and directly addresses the QUESTION.
        
        QUESTION:
        {query}
        
        ANSWER:
        {answer}
        
        Is the answer relevant?
        Reply ONLY with a score from 0.0 (Irrelevant) to 1.0 (Very Relevant). Do not give explanation.
        Score: [/INST]"""
        
        return self._get_llm_score(prompt)

    def _get_llm_score(self, prompt):
        try:
            # Generate 1 token saja (angkanya)
            output = self.llm(prompt, max_tokens=5, temperature=0.1, echo=False)
            text = output['choices'][0]['text'].strip()
            
            # Bersihkan output (ambil angka pertama yg muncul)
            match = re.search(r"0\.\d+|1\.0|0|1", text)
            if match:
                return float(match.group())
            return 0.5 # Default jika format salah
        except:
            return 0.0