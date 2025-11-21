import torch
import pandas as pd
import math
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.core.config import DEVICE

class TripletExtractor:
    def __init__(self, model_name='Babelscape/rebel-large'):
        """
        Inisialisasi model REBEL untuk ekstraksi relasi.
        """
        self.device = DEVICE
        print(f"[Extractor] Menggunakan device: {self.device}")
        print(f"[Extractor] Memuat model {model_name}...")
        
        # Load tokenizer dan model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        print("[Extractor] Model berhasil dimuat.")
        
    def _extract_triplets_from_text(self, text):
        """
        Fungsi helper untuk parsing output raw REBEL
        """
        triplets = []
        relation, subject, object_ = '', '', ''
        text = text.strip()
        current = 'x'
        
        # Membersihkan token spesial dari output model
        text_replace = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "")
        
        for token in text_replace.split():
            if token == "<triplets>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't': subject += ' ' + token
                elif current == 's': object_ += ' ' + token
                elif current == 'o': relation += ' ' + token
        
        # Menangkap triplets terakhir yang tersisa di buffer
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail': object_.strip()})
        return triplets
    
    def process_batch(self, data_chunks, batch_size=4):
        """
        Memproses list of chunks secara batch.
        data_chunks: list of dict (hasil dari preprocessing)
        """
        results = []
        total_chunks = len(data_chunks)
        num_batches = math.ceil(total_chunks / batch_size)
        print(f"[Extractor] Memulai ekstraksi relasi untuk {total_chunks} chunks...")
        
        # Loop per batch
        for i in tqdm(range(0, total_chunks, batch_size), total=num_batches, desc="Ekstraksi"):
            batch = data_chunks[i : i + batch_size]
            texts = [item['text'] for item in batch] # Ambil teksnya
            
            # Tokenisasi
            inputs = self.tokenizer(
                texts,
                max_length=256,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=256,
                    length_penalty=0,
                    num_beams=3, # agar hasil lebih variatif/akurat
                    num_return_sequences=1
                )
            
            # decode hasil
            decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)
            
            # Parsing dan mapping kembali ke id chunk
            for idx, pred_text in enumerate(decoded_preds):
                extracted_triplets = self._extract_triplets_from_text(pred_text)
                source_meta = batch[idx] # Metadata chunk asli
                
                for triplet in extracted_triplets:
                    results.append({
                        'chunk_id' : source_meta['id'],
                        'source_title' : source_meta['title'],
                        'head' : triplet['head'],
                        'relation' : triplet['type'],
                        'tail' : triplet['tail'],
                    })           
        return pd.DataFrame(results)
                    