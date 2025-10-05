import streamlit as st
from sentence_transformers import SentenceTransformer
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
        with st.spinner("🔄 Loading multilingual AI model..."):
            try:
                # MULTILINGUAL model - smaller version (50+ languages)
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
            except Exception as e:
                st.error(f"Model error: {e}")
                raise
        
        self.ocr_reader = None
        self.plagiarism_threshold = 0.70
        self.exact_match_threshold = 0.95
    
    def load_ocr(self):
        """Load OCR only when user uploads an image"""
        if self.ocr_reader is None:
            try:
                import easyocr
                with st.spinner("🔄 Loading OCR (first time only)..."):
                    # Load only essential languages
                    self.ocr_reader = easyocr.Reader(['en', 'hi', 'ar', 'zh_sim'], gpu=False)
                st.success("✅ OCR loaded!")
            except Exception as e:
                st.error(f"OCR not available: {e}")
                return None
        return self.ocr_reader
    
    def extract_text_from_image(self, image):
        try:
            from PIL import Image
            reader = self.load_ocr()
            if reader is None:
                st.warning("OCR not available. Please paste text manually.")
                return ""
            
            results = reader.readtext(np.array(image), detail=0)
            return '\n'.join(results) if results else ""
        except Exception as e:
            st.error(f"Image extraction error: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
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
    
    def extract_text_from_file(self, uploaded_file):
        if not uploaded_file:
            return ""
        
        try:
            file_type = uploaded_file.type
            
            if file_type == "application/pdf":
                return self.extract_text_from_pdf(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self.extract_text_from_docx(uploaded_file)
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                from PIL import Image
                return self.extract_text_from_image(Image.open(uploaded_file))
            else:
                st.warning(f"Unsupported: {file_type}")
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
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def detect_plagiarism(self, submitted_text, reference_texts):
        try:
            submitted_sentences = sent_tokenize(submitted_text)
        except:
            submitted_sentences = [s.strip() for s in submitted_text.split('.') if s.strip()]
        
        if not submitted_sentences or not reference_texts:
            return self._empty_result()
        
        all_ref_sentences = []
        sentence_sources = []
        
        for idx, ref_text in enumerate(reference_texts):
            try:
                ref_sents = sent_tokenize(ref_text)
            except:
                ref_sents = [s.strip() for s in ref_text.split('.') if s.strip()]
            
            for sent in ref_sents:
                if len(sent.strip()) > 15:
                    all_ref_sentences.append(sent.strip())
                    sentence_sources.append(f"Reference {idx + 1}")
        
        if not all_ref_sentences:
            return self._empty_result()
        
        st.info(f"🔍 Analyzing {len(submitted_sentences)} sentences...")
        
        try:
            with st.spinner("Encoding (multilingual)..."):
                submitted_emb = self.model.encode(submitted_sentences, show_progress_bar=False)
                reference_emb = self.model.encode(all_ref_sentences, show_progress_bar=False)
        except Exception as e:
            st.error(f"Encoding error: {e}")
            return self._empty_result()
        
        plagiarism_details = []
        plagiarized_count = 0
        exact_copy_count = 0
        
        progress_bar = st.progress(0)
        
        for i, sent in enumerate(submitted_sentences):
            try:
                progress_bar.progress((i + 1) / len(submitted_sentences))
                
                similarities = cosine_similarity([submitted_emb[i]], reference_emb)[0]
                best_idx = int(np.argmax(similarities))
                sem_score = float(similarities[best_idx])
                
                matched_sentence = all_ref_sentences[best_idx]
                source = sentence_sources[best_idx]
                
                exact_match_score = self.calculate_exact_match(sent, matched_sentence)
                ngram_score = self.calculate_ngram_similarity(sent, matched_sentence, n=3)
                tfidf_score = self.calculate_tfidf_similarity(sent, matched_sentence)
                
                combined_score = (
                    sem_score * 0.40 +
                    exact_match_score * 0.30 +
                    ngram_score * 0.20 +
                    tfidf_score * 0.10
                )
                
                is_exact_copy = combined_score >= self.exact_match_threshold
                is_plagiarized = combined_score >= self.plagiarism_threshold
                
                if is_plagiarized:
                    plagiarized_count += 1
                    if is_exact_copy:
                        exact_copy_count += 1
                
                if is_exact_copy:
                    category = "EXACT COPY"
                    color = "🔴"
                elif is_plagiarized:
                    category = "PLAGIARIZED"
                    color = "🟠"
                else:
                    category = "ORIGINAL"
                    color = "🟢"
                
                plagiarism_details.append({
                    'sentence_number': i + 1,
                    'sentence': sent.strip(),
                    'category': category,
                    'color': color,
                    'is_plagiarized': is_plagiarized,
                    'is_exact_copy': is_exact_copy,
                    'combined_score': float(combined_score),
                    'semantic_score': float(sem_score),
                    'exact_match_score': float(exact_match_score),
                    'ngram_score': float(ngram_score),
                    'tfidf_score': float(tfidf_score),
                    'matched_text': matched_sentence,
                    'source': source
                })
            
            except Exception as e:
                continue
        
        progress_bar.empty()
        
        total_sents = len(submitted_sentences)
        original_count = total_sents - plagiarized_count
        
        overall_plagiarism = (plagiarized_count / total_sents * 100) if total_sents > 0 else 0
        
        return {
            'overall_plagiarism': float(overall_plagiarism),
            'originality': float(100 - overall_plagiarism),
            'total_sentences': int(total_sents),
            'plagiarized_sentences': int(plagiarized_count),
            'exact_copies': int(exact_copy_count),
            'original_sentences': int(original_count),
            'details': plagiarism_details
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
    st.markdown("### 50+ Languages • OCR Support • Free")
    
    with st.sidebar:
        st.header("🎯 Features")
        st.success("✅ Multilingual (50+ langs)")
        st.success("✅ OCR (on-demand)")
        st.success("✅ PDF/DOCX/Images")
        st.success("✅ 4 Detection Methods")
        
        st.divider()
        
        st.info("""
        **Supported Languages:**
        English, Spanish, French, German, 
        Italian, Portuguese, Dutch, Polish,
        Russian, Arabic, Hindi, Chinese,
        Japanese, Korean, and 35+ more!
        """)
    
    detector = load_detector()
    
    st.header("Plagiarism Detection")
    st.info("📌 Works with ANY language • Upload submission + references")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📄 Student Submission")
        
        submission_file = st.file_uploader(
            "Upload (PDF/DOCX/Image)", 
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], 
            key="sub"
        )
        
        submission_text = ""
        
        if submission_file:
            with st.spinner("Extracting..."):
                submission_text = detector.extract_text_from_file(submission_file)
            
            if submission_text:
                st.success(f"✅ {len(submission_text)} characters")
                
                if submission_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                    with st.expander("View Image"):
                        st.image(submission_file)
        
        submission_text = st.text_area(
            "Or paste text (any language)",
            value=submission_text,
            height=300,
            placeholder="Paste text in any language..."
        )
    
    with col2:
        st.subheader("📚 References")
        
        num_refs = st.number_input("Number", 1, 5, 2)
        
        reference_texts = []
        
        for i in range(num_refs):
            with st.expander(f"Reference {i+1}", expanded=(i==0)):
                ref_file = st.file_uploader(
                    f"Upload",
                    type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
                    key=f"ref{i}"
                )
                
                ref_text = ""
                
                if ref_file:
                    with st.spinner("Extracting..."):
                        ref_text = detector.extract_text_from_file(ref_file)
                    if ref_text:
                        st.success(f"✅ {len(ref_text)} chars")
                
                ref_text = st.text_area(
                    "Or paste",
                    value=ref_text,
                    height=100,
                    key=f"rt{i}"
                )
                
                if ref_text and ref_text.strip():
                    reference_texts.append(ref_text.strip())
    
    if st.button("🔍 DETECT PLAGIARISM", type="primary", use_container_width=True):
        if not submission_text or not submission_text.strip():
            st.error("❌ Provide submission")
        elif not reference_texts:
            st.error("❌ Provide references")
        else:
            try:
                results = detector.detect_plagiarism(submission_text, reference_texts)
                
                st.success("✅ ANALYSIS COMPLETE!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                plag_pct = results['overall_plagiarism']
                orig_pct = results['originality']
                
                with col1:
                    if plag_pct > 30:
                        st.metric("Plagiarism", f"🔴 {plag_pct:.1f}%")
                    elif plag_pct > 15:
                        st.metric("Plagiarism", f"🟠 {plag_pct:.1f}%")
                    else:
                        st.metric("Plagiarism", f"🟢 {plag_pct:.1f}%")
                
                with col2:
                    st.metric("Originality", f"{orig_pct:.1f}%")
                
                with col3:
                    st.metric("Plagiarized", f"{results['plagiarized_sentences']}/{results['total_sentences']}")
                
                with col4:
                    st.metric("Exact Copies", f"🔴 {results['exact_copies']}")
                
                st.divider()
                
                if plag_pct < 10:
                    st.success("✅ **EXCELLENT**")
                elif plag_pct < 25:
                    st.warning("⚠️ **ACCEPTABLE**")
                else:
                    st.error("🚨 **CONCERNING**")
                
                st.divider()
                
                st.subheader("📊 Detailed Analysis")
                
                exact_copies = [d for d in results['details'] if d['is_exact_copy']]
                plagiarized = [d for d in results['details'] if d['is_plagiarized'] and not d['is_exact_copy']]
                original = [d for d in results['details'] if not d['is_plagiarized']]
                
                if exact_copies:
                    st.error(f"🔴 **EXACT COPIES ({len(exact_copies)})**")
                    for item in exact_copies[:5]:
                        st.markdown(f"**Sentence {item['sentence_number']}** - {item['combined_score']:.1%}")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(item['sentence'])
                        with col2:
                            st.warning(f"{item['source']}: {item['matched_text']}")
                        st.divider()
                
                if plagiarized:
                    with st.expander(f"🟠 Plagiarized ({len(plagiarized)})"):
                        for item in plagiarized[:10]:
                            st.markdown(f"**{item['sentence_number']}.** {item['sentence']}")
                            st.caption(f"Matched: {item['matched_text'][:100]}...")
                
                with st.expander(f"🟢 Original ({len(original)})"):
                    for item in original[:10]:
                        st.markdown(f"**{item['sentence_number']}.** {item['sentence']}")
            
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
