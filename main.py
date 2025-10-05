import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
import PyPDF2
from docx import Document
import tempfile
import os
import ssl
import re
from difflib import SequenceMatcher
import easyocr
from PIL import Image

st.set_page_config(page_title="Multilingual Plagiarism Detector", layout="wide", page_icon="🔍")

@st.cache_resource
def download_nltk_data():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass

download_nltk_data()

class MultilingualPlagiarismDetector:
    def __init__(self):
        with st.spinner("🔄 Loading AI models..."):
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')
            self.paraphrase_model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device='cpu')
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
        self.ocr_reader = None
        self.plagiarism_threshold = 0.70
        self.exact_match_threshold = 0.95
    
    def load_ocr(self):
        if self.ocr_reader is None:
            with st.spinner("🔄 Loading OCR..."):
                self.ocr_reader = easyocr.Reader(['en', 'hi', 'ar', 'zh_sim'], gpu=False)
        return self.ocr_reader
    
    def extract_text_from_pdf(self, pdf_bytes):
        try:
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"PDF error: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_bytes):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(docx_bytes)
                tmp.flush()
                tmp_path = tmp.name
            
            doc = Document(tmp_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text])
            
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            return text.strip()
        except Exception as e:
            st.error(f"DOCX error: {str(e)}")
            return ""
    
    def extract_text_from_image(self, image_bytes):
        try:
            import io
            reader = self.load_ocr()
            image = Image.open(io.BytesIO(image_bytes))
            results = reader.readtext(np.array(image), detail=0)
            return '\n'.join(results) if results else ""
        except Exception as e:
            st.error(f"OCR error: {str(e)}")
            return ""
    
    def extract_text(self, uploaded_file):
        """Extract text from uploaded file"""
        if not uploaded_file:
            return ""
        
        try:
            # Read file bytes once
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            if file_type == "application/pdf":
                return self.extract_text_from_pdf(file_bytes)
            elif "wordprocessing" in file_type:
                return self.extract_text_from_docx(file_bytes)
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                return self.extract_text_from_image(file_bytes)
            else:
                st.warning(f"Unsupported type: {file_type}")
                return ""
        except Exception as e:
            st.error(f"Extraction failed: {str(e)}")
            return ""
    
    def calculate_exact_match(self, text1, text2):
        try:
            text1_clean = re.sub(r'[^\w\s]', '', text1.lower())
            text2_clean = re.sub(r'[^\w\s]', '', text2.lower())
            matcher = SequenceMatcher(None, text1_clean, text2_clean)
            return matcher.ratio()
        except:
            return 0.0
    
    def calculate_ngram_similarity(self, text1, text2, n=3):
        try:
            chars1 = list(text1.lower().replace(' ', ''))
            chars2 = list(text2.lower().replace(' ', ''))
            if len(chars1) < n or len(chars2) < n:
                return 0.0
            ngrams1 = set(tuple(chars1[i:i+n]) for i in range(len(chars1)-n+1))
            ngrams2 = set(tuple(chars2[i:i+n]) for i in range(len(chars2)-n+1))
            if not ngrams1 or not ngrams2:
                return 0.0
            intersection = len(ngrams1.intersection(ngrams2))
            union = len(ngrams1.union(ngrams2))
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def calculate_tfidf_similarity(self, text1, text2):
        try:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def detect_plagiarism_advanced(self, submission, references):
        try:
            sub_sents = sent_tokenize(submission)
        except:
            sub_sents = [s.strip() for s in submission.split('.') if s.strip()]
        
        if not sub_sents or not references:
            return self._empty_result()
        
        ref_sents = []
        sources = []
        for idx, ref in enumerate(references):
            try:
                sents = sent_tokenize(ref)
            except:
                sents = [s.strip() for s in ref.split('.') if s.strip()]
            for s in sents:
                if len(s) > 15:
                    ref_sents.append(s)
                    sources.append(f"Ref {idx+1}")
        
        if not ref_sents:
            return self._empty_result()
        
        st.info(f"🔍 Analyzing {len(sub_sents)} sentences...")
        
        with st.spinner("Encoding..."):
            sub_emb_sem = self.semantic_model.encode(sub_sents, show_progress_bar=False)
            ref_emb_sem = self.semantic_model.encode(ref_sents, show_progress_bar=False)
            sub_emb_para = self.paraphrase_model.encode(sub_sents, show_progress_bar=False)
            ref_emb_para = self.paraphrase_model.encode(ref_sents, show_progress_bar=False)
        
        details = []
        plag_count = 0
        exact_count = 0
        progress = st.progress(0)
        
        for i, sent in enumerate(sub_sents):
            progress.progress((i+1)/len(sub_sents))
            
            sem_sims = cosine_similarity([sub_emb_sem[i]], ref_emb_sem)[0]
            para_sims = cosine_similarity([sub_emb_para[i]], ref_emb_para)[0]
            
            sem_idx = int(np.argmax(sem_sims))
            para_idx = int(np.argmax(para_sims))
            
            sem_score = float(sem_sims[sem_idx])
            para_score = float(para_sims[para_idx])
            
            if sem_score > para_score:
                matched_idx = sem_idx
                ai_score = sem_score
            else:
                matched_idx = para_idx
                ai_score = para_score
            
            matched_sent = ref_sents[matched_idx]
            source = sources[matched_idx]
            
            exact_score = self.calculate_exact_match(sent, matched_sent)
            ngram_score = self.calculate_ngram_similarity(sent, matched_sent)
            tfidf_score = self.calculate_tfidf_similarity(sent, matched_sent)
            
            combined = (ai_score * 0.35 + exact_score * 0.30 + ngram_score * 0.20 + tfidf_score * 0.15)
            
            is_exact = combined >= 0.95
            is_plag = combined >= 0.70
            
            if is_plag:
                plag_count += 1
                if is_exact:
                    exact_count += 1
            
            details.append({
                'sentence_number': i+1,
                'sentence': sent,
                'is_plagiarized': is_plag,
                'is_exact_copy': is_exact,
                'combined_score': float(combined),
                'matched_text': matched_sent,
                'source': source
            })
        
        progress.empty()
        
        total = len(sub_sents)
        plag_pct = (plag_count / total * 100) if total > 0 else 0
        
        return {
            'overall_plagiarism': float(plag_pct),
            'originality': float(100 - plag_pct),
            'total_sentences': int(total),
            'plagiarized_sentences': int(plag_count),
            'exact_copies': int(exact_count),
            'original_sentences': int(total - plag_count),
            'details': details
        }
    
    def _empty_result(self):
        return {
            'overall_plagiarism': 0.0,
            'originality': 100.0,
            'total_sentences': 0,
            'plagiarized_sentences': 0,
            'exact_copies': 0,
            'original_sentences': 0,
            'details': []
        }

@st.cache_resource(show_spinner=False)
def load_detector():
    return MultilingualPlagiarismDetector()

def main():
    st.title("🌍 Multilingual Plagiarism Detector")
    st.markdown("### 3 AI Models • OCR • 100+ Languages")
    
    detector = load_detector()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Submission")
        sub_file = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key="sub")
        
        sub_text = ""
        if sub_file:
            with st.spinner("Extracting..."):
                sub_text = detector.extract_text(sub_file)
            if sub_text:
                st.success(f"✅ {len(sub_text)} chars")
        
        sub_input = st.text_area("Text", value=sub_text, height=300)
    
    with col2:
        st.subheader("📚 References")
        num_refs = st.number_input("Number", 1, 5, 1)
        
        ref_texts = []
        
        for i in range(num_refs):
            st.markdown(f"**Ref {i+1}**")
            ref_file = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key=f"ref{i}")
            
            ref_text = ""
            if ref_file:
                with st.spinner("Extracting..."):
                    ref_text = detector.extract_text(ref_file)
                if ref_text:
                    st.success(f"✅ {len(ref_text)} chars")
                    ref_texts.append(ref_text)
            
            ref_input = st.text_area("Text", value=ref_text, height=100, key=f"rt{i}")
            
            if ref_input and ref_input.strip() and not ref_file:
                ref_texts.append(ref_input.strip())
    
    if st.button("🔍 DETECT", type="primary"):
        if not sub_input or not sub_input.strip():
            st.error("❌ Provide submission")
        elif not ref_texts:
            st.error("❌ Provide references")
        else:
            results = detector.detect_plagiarism_advanced(sub_input, ref_texts)
            
            st.success("✅ DONE!")
            
            plag = results['overall_plagiarism']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Plagiarism", f"{'🔴' if plag>30 else '🟠' if plag>15 else '🟢'} {plag:.1f}%")
            with col2:
                st.metric("Originality", f"{results['originality']:.1f}%")
            with col3:
                st.metric("Plagiarized", f"{results['plagiarized_sentences']}/{results['total_sentences']}")
            with col4:
                st.metric("Exact", f"{results['exact_copies']}")
            
            st.divider()
            
            exact = [d for d in results['details'] if d['is_exact_copy']]
            if exact:
                st.error(f"🔴 {len(exact)} EXACT COPIES")
                for item in exact[:3]:
                    st.markdown(f"**{item['sentence_number']}.** {item['sentence']}")
                    st.caption(f"From {item['source']}: {item['matched_text'][:100]}...")
                    st.divider()

if __name__ == "__main__":
    main()
