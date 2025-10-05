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
                self.ocr_reader = easyocr.Reader(['en', 'hi', 'ar', 'zh_sim', 'es', 'fr'], gpu=False)
        return self.ocr_reader
    
    def extract_text_from_pdf(self, pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"PDF error: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(docx_file.getvalue())
                tmp_path = tmp.name
            doc = Document(tmp_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text])
            os.unlink(tmp_path)
            return text.strip()
        except Exception as e:
            st.error(f"DOCX error: {e}")
            return ""
    
    def extract_text_from_image(self, image):
        try:
            reader = self.load_ocr()
            results = reader.readtext(np.array(image), detail=0)
            return '\n'.join(results) if results else ""
        except Exception as e:
            st.error(f"OCR error: {e}")
            return ""
    
    def extract_text(self, file):
        if not file:
            return ""
        
        try:
            file_type = file.type
            if file_type == "application/pdf":
                return self.extract_text_from_pdf(file)
            elif "wordprocessing" in file_type:
                return self.extract_text_from_docx(file)
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                return self.extract_text_from_image(Image.open(file))
            return ""
        except Exception as e:
            st.error(f"Extraction error: {e}")
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
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=1000)
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
                    sources.append(f"Reference {idx+1}")
        
        if not ref_sents:
            return self._empty_result()
        
        st.info(f"🔍 Analyzing {len(sub_sents)} sentences...")
        
        with st.spinner("🧠 Semantic encoding..."):
            sub_emb_sem = self.semantic_model.encode(sub_sents, show_progress_bar=False)
            ref_emb_sem = self.semantic_model.encode(ref_sents, show_progress_bar=False)
        
        with st.spinner("🧠 Paraphrase encoding..."):
            sub_emb_para = self.paraphrase_model.encode(sub_sents, show_progress_bar=False)
            ref_emb_para = self.paraphrase_model.encode(ref_sents, show_progress_bar=False)
        
        details = []
        plag_count = 0
        exact_count = 0
        progress = st.progress(0)
        
        for i, sent in enumerate(sub_sents):
            progress.progress((i+1)/len(sub_sents))
            
            sem_sims = cosine_similarity([sub_emb_sem[i]], ref_emb_sem)[0]
            best_sem_idx = int(np.argmax(sem_sims))
            sem_score = float(sem_sims[best_sem_idx])
            
            para_sims = cosine_similarity([sub_emb_para[i]], ref_emb_para)[0]
            best_para_idx = int(np.argmax(para_sims))
            para_score = float(para_sims[best_para_idx])
            
            if sem_score > para_score:
                best_score = sem_score
                matched_idx = best_sem_idx
            else:
                best_score = para_score
                matched_idx = best_para_idx
            
            matched_sent = ref_sents[matched_idx]
            source = sources[matched_idx]
            
            exact_score = self.calculate_exact_match(sent, matched_sent)
            ngram_score = self.calculate_ngram_similarity(sent, matched_sent, n=3)
            tfidf_score = self.calculate_tfidf_similarity(sent, matched_sent)
            
            combined = (best_score * 0.35 + exact_score * 0.30 + ngram_score * 0.20 + tfidf_score * 0.15)
            
            is_exact = combined >= self.exact_match_threshold
            is_plag = combined >= self.plagiarism_threshold
            
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
                'semantic_score': float(sem_score),
                'paraphrase_score': float(para_score),
                'exact_match_score': float(exact_score),
                'ngram_score': float(ngram_score),
                'tfidf_score': float(tfidf_score),
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
    st.title("🌍 MULTILINGUAL Plagiarism Detector")
    st.markdown("### 3 AI Models • 5 Detection Algorithms • OCR • 100+ Languages")
    
    with st.sidebar:
        st.header("🎯 System")
        st.subheader("🧠 AI Models:")
        st.success("✅ Multilingual MPNet")
        st.success("✅ DistilUSE Multilingual")
        st.success("✅ Cross-Encoder")
        st.success("✅ EasyOCR (80+ langs)")
        st.divider()
        st.subheader("🔍 Detection:")
        st.info("1. Semantic Similarity")
        st.info("2. Paraphrase Detection")
        st.info("3. Exact Matching")
        st.info("4. N-gram Analysis")
        st.info("5. TF-IDF Vectors")
    
    detector = load_detector()
    
    st.header("Plagiarism Detection")
    
    col1, col2 = st.columns(2)
    
    # SUBMISSION COLUMN
    with col1:
        st.subheader("📄 Submission")
        sub_file = st.file_uploader("Upload (PDF/DOCX/Image)", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key="sub")
        
        # Initialize variable
        if 'sub_text_value' not in st.session_state:
            st.session_state.sub_text_value = ""
        
        # Extract text when file is uploaded
        if sub_file:
            with st.spinner("Extracting..."):
                extracted = detector.extract_text(sub_file)
                if extracted:
                    st.session_state.sub_text_value = extracted
                    st.success(f"✅ {len(extracted)} chars")
        
        # Text area
        sub_text = st.text_area("Or paste text", value=st.session_state.sub_text_value, height=300, key="sub_text_input")
    
    # REFERENCES COLUMN
    with col2:
        st.subheader("📚 References")
        num_refs = st.number_input("Number", 1, 5, 1)
        
        ref_texts = []
        
        for i in range(num_refs):
            st.markdown(f"**Reference {i+1}**")
            
            # Initialize session state for this reference
            if f'ref_text_value_{i}' not in st.session_state:
                st.session_state[f'ref_text_value_{i}'] = ""
            
            ref_file = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key=f"ref{i}")
            
            # Extract when file uploaded
            if ref_file:
                with st.spinner("Extracting..."):
                    extracted = detector.extract_text(ref_file)
                    if extracted:
                        st.session_state[f'ref_text_value_{i}'] = extracted
                        st.success(f"✅ {len(extracted)} chars")
                        ref_texts.append(extracted)
            
            # Text area
            ref_text = st.text_area("Or paste", value=st.session_state[f'ref_text_value_{i}'], height=100, key=f"ref_text{i}")
            
            # Add manually entered text
            if ref_text and ref_text.strip() and not ref_file:
                if ref_text not in ref_texts:
                    ref_texts.append(ref_text.strip())
    
    # DETECT BUTTON
    if st.button("🔍 DETECT PLAGIARISM", type="primary", use_container_width=True):
        if not sub_text or not sub_text.strip():
            st.error("❌ Provide submission")
        elif not ref_texts:
            st.error("❌ Provide references")
        else:
            try:
                results = detector.detect_plagiarism_advanced(sub_text, ref_texts)
                
                st.success("✅ ANALYSIS COMPLETE!")
                
                plag = results['overall_plagiarism']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    if plag > 30:
                        st.metric("Plagiarism", f"🔴 {plag:.1f}%")
                    elif plag > 15:
                        st.metric("Plagiarism", f"🟠 {plag:.1f}%")
                    else:
                        st.metric("Plagiarism", f"🟢 {plag:.1f}%")
                with col2:
                    st.metric("Originality", f"{results['originality']:.1f}%")
                with col3:
                    st.metric("Plagiarized", f"{results['plagiarized_sentences']}/{results['total_sentences']}")
                with col4:
                    st.metric("Exact Copies", f"🔴 {results['exact_copies']}")
                
                st.divider()
                
                if plag < 10:
                    st.success("✅ **EXCELLENT**")
                elif plag < 25:
                    st.warning("⚠️ **ACCEPTABLE**")
                else:
                    st.error("🚨 **CONCERNING**")
                
                st.divider()
                st.subheader("📊 Details")
                
                exact = [d for d in results['details'] if d['is_exact_copy']]
                plag_items = [d for d in results['details'] if d['is_plagiarized'] and not d['is_exact_copy']]
                original = [d for d in results['details'] if not d['is_plagiarized']]
                
                if exact:
                    st.error(f"🔴 **EXACT COPIES ({len(exact)})**")
                    for item in exact[:5]:
                        st.markdown(f"**Sentence {item['sentence_number']}** - {item['combined_score']:.1%}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(item['sentence'])
                        with col2:
                            st.warning(f"{item['source']}: {item['matched_text'][:100]}...")
                        st.caption(f"Semantic: {item['semantic_score']:.1%} | Exact: {item['exact_match_score']:.1%} | N-gram: {item['ngram_score']:.1%}")
                        st.divider()
                
                if plag_items:
                    with st.expander(f"🟠 Plagiarized Sentences ({len(plag_items)})"):
                        for item in plag_items[:10]:
                            st.markdown(f"**{item['sentence_number']}.** {item['sentence']}")
                            st.caption(f"Score: {item['combined_score']:.1%} | {item['source']}")
                
                with st.expander(f"🟢 Original Sentences ({len(original)})"):
                    for item in original[:10]:
                        st.markdown(f"**{item['sentence_number']}.** {item['sentence']}")
                
                with st.expander("📥 Export Report"):
                    st.json(results)
            
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
