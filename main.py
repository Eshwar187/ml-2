import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from collections import Counter
import hashlib
import easyocr
from PIL import Image
import io
import pandas as pd
import PyPDF2
from docx import Document
import tempfile
import os

# Configure page FIRST
st.set_page_config(page_title="AI Plagiarism Detector", layout="wide", page_icon="🔍")

# NLTK setup
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('stopwords', quiet=True)

download_nltk_data()

# Stylometry class
class StylometricAnalyzer:
    def __init__(self):
        self.features = {}
    
    def extract_stylometric_features(self, text):
        if not text or len(text.strip()) < 10:
            return self._empty_features()
        
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            pos_tags = pos_tag(word_tokenize(text))
            
            return {
                'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
                'sentence_length_std': np.std([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
                'type_token_ratio': len(set(words)) / len(words) if words else 0,
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'punctuation_density': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0,
                'capital_letter_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'pos_distribution': dict(Counter([tag for _, tag in pos_tags]))
            }
        except:
            return self._empty_features()
    
    def _empty_features(self):
        return {
            'avg_sentence_length': 0,
            'sentence_length_std': 0,
            'type_token_ratio': 0,
            'avg_word_length': 0,
            'punctuation_density': 0,
            'capital_letter_ratio': 0,
            'pos_distribution': {}
        }
    
    def calculate_style_similarity(self, features1, features2):
        f1 = np.array([
            features1['avg_sentence_length'],
            features1['sentence_length_std'],
            features1['type_token_ratio'],
            features1['avg_word_length'],
            features1['punctuation_density'],
            features1['capital_letter_ratio']
        ]).reshape(1, -1)
        
        f2 = np.array([
            features2['avg_sentence_length'],
            features2['sentence_length_std'],
            features2['type_token_ratio'],
            features2['avg_word_length'],
            features2['punctuation_density'],
            features2['capital_letter_ratio']
        ]).reshape(1, -1)
        
        return 1 / (1 + np.linalg.norm(f1 - f2))

# Fingerprinting class
class DocumentFingerprinter:
    def __init__(self, k=5, window_size=4):
        self.k = k
        self.window_size = window_size
    
    def create_kgrams(self, text):
        text = text.lower().replace(' ', '')
        return [text[i:i+self.k] for i in range(len(text) - self.k + 1)]
    
    def hash_kgrams(self, kgrams):
        return [int(hashlib.md5(kg.encode()).hexdigest(), 16) % (10**8) for kg in kgrams]
    
    def winnow(self, hashes):
        if len(hashes) < self.window_size:
            return set((i, h) for i, h in enumerate(hashes))
        
        fingerprints = set()
        for i in range(len(hashes) - self.window_size + 1):
            window = hashes[i:i + self.window_size]
            min_hash = min(window)
            fingerprints.add((i + window.index(min_hash), min_hash))
        return fingerprints
    
    def get_fingerprint(self, text):
        if not text or len(text) < self.k:
            return set()
        kgrams = self.create_kgrams(text)
        if not kgrams:
            return set()
        hashes = self.hash_kgrams(kgrams)
        return self.winnow(hashes)
    
    def compare_fingerprints(self, fp1, fp2):
        if not fp1 or not fp2:
            return 0.0
        hashes1 = set(h for _, h in fp1)
        hashes2 = set(h for _, h in fp2)
        intersection = len(hashes1.intersection(hashes2))
        union = len(hashes1.union(hashes2))
        return intersection / union if union > 0 else 0.0

# Novelty detector
class NoveltyDetector:
    def __init__(self, model):
        self.model = model
        self.novelty_threshold = 0.75
    
    def detect_novel_sentences(self, target_text, source_texts):
        target_sentences = sent_tokenize(target_text)
        all_source_sentences = []
        for source in source_texts:
            all_source_sentences.extend(sent_tokenize(source))
        
        if not all_source_sentences:
            return [(s, 1.0, "Novel", 0.0) for s in target_sentences]
        
        target_emb = self.model.encode(target_sentences, show_progress_bar=False)
        source_emb = self.model.encode(all_source_sentences, show_progress_bar=False)
        
        results = []
        for sentence, emb in zip(target_sentences, target_emb):
            sims = cosine_similarity([emb], source_emb)[0]
            max_sim = np.max(sims)
            novelty = 1 - max_sim
            status = "Novel" if novelty > (1 - self.novelty_threshold) else "Redundant"
            results.append((sentence, novelty, status, max_sim))
        
        return results

# Main system
class PlagiarismDetectionSystem:
    def __init__(self, performance_mode="balanced"):
        self.embedding_models = {
            "fast": "paraphrase-multilingual-MiniLM-L12-v2",
            "balanced": "paraphrase-multilingual-mpnet-base-v2",
            "best": "paraphrase-multilingual-mpnet-base-v2"
        }
        
        with st.spinner("🔄 Loading AI models..."):
            self.embedding_model = SentenceTransformer(
                self.embedding_models[performance_mode],
                device="cpu"
            )
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu",
                max_length=512
            )
        
        self.stylometric_analyzer = StylometricAnalyzer()
        self.fingerprinter = DocumentFingerprinter()
        self.novelty_detector = NoveltyDetector(self.embedding_model)
        self.ocr_reader = None
    
    def load_ocr(self):
        if self.ocr_reader is None:
            with st.spinner("🔄 Loading OCR..."):
                self.ocr_reader = easyocr.Reader(['en', 'hi', 'ar', 'zh_sim'], gpu=False)
        return self.ocr_reader
    
    def extract_text_from_image(self, image):
        try:
            reader = self.load_ocr()
            results = reader.readtext(np.array(image), detail=0)
            return '\n'.join(results) if results else ""
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""
    
    def extract_text_from_pdf(self, pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            return '\n'.join(page.extract_text() or "" for page in reader.pages).strip()
        except Exception as e:
            st.error(f"PDF Error: {str(e)}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(docx_file.getvalue())
                tmp_path = tmp.name
            
            doc = Document(tmp_path)
            text = '\n'.join(p.text for p in doc.paragraphs if p.text).strip()
            os.unlink(tmp_path)
            return text
        except Exception as e:
            st.error(f"DOCX Error: {str(e)}")
            return ""
    
    def extract_text_from_file(self, uploaded_file):
        """Universal file handler"""
        if not uploaded_file:
            return ""
        
        try:
            file_type = uploaded_file.type
            
            if file_type == "application/pdf":
                return self.extract_text_from_pdf(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self.extract_text_from_docx(uploaded_file)
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                return self.extract_text_from_image(Image.open(uploaded_file))
            else:
                st.warning(f"Unsupported: {file_type}")
                return ""
        except Exception as e:
            st.error(f"Extraction error: {str(e)}")
            return ""
    
    def compare_multiple_assignments(self, assignments_dict):
        names = list(assignments_dict.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        details = {}
        
        progress = st.progress(0)
        total = (n * (n - 1)) // 2
        current = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                text1 = assignments_dict[names[i]]
                text2 = assignments_dict[names[j]]
                
                emb1 = self.embedding_model.encode([text1], show_progress_bar=False)[0]
                emb2 = self.embedding_model.encode([text2], show_progress_bar=False)[0]
                sem_sim = cosine_similarity([emb1], [emb2])[0][0]
                
                fp1 = self.fingerprinter.get_fingerprint(text1)
                fp2 = self.fingerprinter.get_fingerprint(text2)
                fp_sim = self.fingerprinter.compare_fingerprints(fp1, fp2)
                
                combined = sem_sim * 0.6 + fp_sim * 0.4
                
                matrix[i][j] = combined
                matrix[j][i] = combined
                
                details[f"{names[i]} vs {names[j]}"] = {
                    'semantic_similarity': float(sem_sim),
                    'fingerprint_similarity': float(fp_sim),
                    'combined_score': float(combined),
                    'student1': names[i],
                    'student2': names[j]
                }
                
                current += 1
                progress.progress(current / total)
        
        progress.empty()
        return matrix, details, names
    
    def analyze_plagiarism(self, submitted_text, reference_texts, student_history=None):
        results = {
            'semantic_similarity': {},
            'stylometric_analysis': {},
            'fingerprint_matching': {},
            'novelty_detection': {},
            'intrinsic_detection': {},
            'overall_score': 0.0
        }
        
        # Semantic
        sub_emb = self.embedding_model.encode([submitted_text], show_progress_bar=False)[0]
        
        for i, ref in enumerate(reference_texts):
            ref_emb = self.embedding_model.encode([ref], show_progress_bar=False)[0]
            bi_sim = cosine_similarity([sub_emb], [ref_emb])[0][0]
            
            try:
                cross_sim = self.cross_encoder.predict([(submitted_text[:512], ref[:512])])[0]
            except:
                cross_sim = bi_sim
            
            combined = bi_sim * 0.6 + float(cross_sim) * 0.4
            
            results['semantic_similarity'][f'Reference_{i+1}'] = {
                'bi_encoder_score': float(bi_sim),
                'cross_encoder_score': float(cross_sim),
                'combined_score': float(combined)
            }
        
        # Stylometric
        sub_feat = self.stylometric_analyzer.extract_stylometric_features(submitted_text)
        
        if student_history:
            hist_feat = self.stylometric_analyzer.extract_stylometric_features(student_history)
            consistency = self.stylometric_analyzer.calculate_style_similarity(sub_feat, hist_feat)
            results['stylometric_analysis']['consistency_with_history'] = float(consistency)
            results['stylometric_analysis']['status'] = 'Consistent' if consistency > 0.7 else 'Inconsistent'
        
        results['stylometric_analysis']['features'] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v 
            for k, v in sub_feat.items() if k != 'pos_distribution'
        }
        
        # Fingerprint
        sub_fp = self.fingerprinter.get_fingerprint(submitted_text)
        for i, ref in enumerate(reference_texts):
            ref_fp = self.fingerprinter.get_fingerprint(ref)
            fp_sim = self.fingerprinter.compare_fingerprints(sub_fp, ref_fp)
            results['fingerprint_matching'][f'Reference_{i+1}'] = float(fp_sim)
        
        # Novelty
        novelty_res = self.novelty_detector.detect_novel_sentences(submitted_text, reference_texts)
        novel_count = sum(1 for _, _, s, _ in novelty_res if s == "Novel")
        total_sent = len(novelty_res)
        
        results['novelty_detection'] = {
            'novel_sentences': novel_count,
            'total_sentences': total_sent,
            'novelty_ratio': novel_count / total_sent if total_sent > 0 else 0,
            'sentence_details': [
                {
                    'sentence': s[:100] + '...' if len(s) > 100 else s,
                    'novelty_score': float(n),
                    'status': st,
                    'max_similarity': float(m)
                }
                for s, n, st, m in novelty_res
            ]
        }
        
        # Intrinsic
        paragraphs = submitted_text.split('\n\n')
        if len(paragraphs) > 1:
            para_feats = [
                self.stylometric_analyzer.extract_stylometric_features(p)
                for p in paragraphs if len(p.strip()) > 50
            ]
            
            if len(para_feats) > 1:
                variations = [
                    float(self.stylometric_analyzer.calculate_style_similarity(para_feats[i], para_feats[i+1]))
                    for i in range(len(para_feats) - 1)
                ]
                
                avg = np.mean(variations)
                results['intrinsic_detection'] = {
                    'avg_style_consistency': float(avg),
                    'style_variations': variations,
                    'suspicious': avg < 0.6
                }
        
        # Overall
        sem_scores = [v['combined_score'] for v in results['semantic_similarity'].values()]
        fp_scores = list(results['fingerprint_matching'].values())
        
        max_sem = max(sem_scores) if sem_scores else 0
        max_fp = max(fp_scores) if fp_scores else 0
        nov_ratio = results['novelty_detection']['novelty_ratio']
        
        overall = max_sem * 0.35 + max_fp * 0.35 + (1 - nov_ratio) * 0.30
        
        results['overall_score'] = float(overall)
        results['risk_level'] = self._get_risk_level(overall)
        results['originality_percentage'] = float((1 - overall) * 100)
        
        return results
    
    def _get_risk_level(self, score):
        if score > 0.75:
            return "🔴 High Risk"
        elif score > 0.50:
            return "🟡 Medium Risk"
        elif score > 0.25:
            return "🟢 Low Risk"
        else:
            return "✅ Minimal Risk"

@st.cache_resource(show_spinner=False)
def load_system(performance_mode):
    return PlagiarismDetectionSystem(performance_mode)

def create_visualization(results):
    if results['semantic_similarity']:
        st.subheader("📊 Semantic Similarity")
        ref_names = list(results['semantic_similarity'].keys())
        bi = [results['semantic_similarity'][r]['bi_encoder_score'] for r in ref_names]
        cross = [results['semantic_similarity'][r]['cross_encoder_score'] for r in ref_names]
        combined = [results['semantic_similarity'][r]['combined_score'] for r in ref_names]
        
        df = pd.DataFrame({
            'Reference': ref_names,
            'Bi-Encoder': bi,
            'Cross-Encoder': cross,
            'Combined': combined
        })
        st.dataframe(df, use_container_width=True)
    
    if results['fingerprint_matching']:
        st.subheader("🔍 Fingerprint Matching")
        fp_data = pd.DataFrame({
            'Reference': list(results['fingerprint_matching'].keys()),
            'Similarity': list(results['fingerprint_matching'].values())
        })
        st.bar_chart(fp_data.set_index('Reference'))
    
    if results['novelty_detection']:
        st.subheader("✨ Novelty Detection")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Novel", results['novelty_detection']['novel_sentences'])
        with col2:
            st.metric("Total", results['novelty_detection']['total_sentences'])
        with col3:
            st.metric("Ratio", f"{results['novelty_detection']['novelty_ratio']:.2%}")

def main():
    st.title("🔍 Advanced Plagiarism Detection System")
    st.markdown("### Multilingual • OCR • PDF/DOCX/Images")
    
    with st.sidebar:
        st.header("⚙️ Settings")
        mode = st.selectbox("Performance", ["fast", "balanced", "best"], index=1)
        st.divider()
        st.success("✅ 80+ Languages")
        st.success("✅ OCR Handwriting")
        st.success("✅ PDF/DOCX/Images")
    
    system = load_system(mode)
    
    tab1, tab2, tab3 = st.tabs(["📝 Originality", "🔄 Compare", "ℹ️ About"])
    
    with tab1:
        st.header("Originality Check")
        st.info("📌 Upload PDF, DOCX, or images with OCR")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Student Submission")
            file1 = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key="f1")
            
            text1 = ""
            if file1:
                with st.spinner("Extracting..."):
                    text1 = system.extract_text_from_file(file1)
                if text1:
                    st.success(f"✅ {len(text1)} chars")
                    if file1.type in ["image/png", "image/jpeg", "image/jpg"]:
                        with st.expander("View"):
                            st.image(file1)
            
            text1 = st.text_area("Or paste", value=text1, height=400, key="t1")
        
        with col2:
            st.subheader("Previous Work")
            file2 = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key="f2")
            
            text2 = ""
            if file2:
                with st.spinner("Extracting..."):
                    text2 = system.extract_text_from_file(file2)
                if text2:
                    st.success(f"✅ {len(text2)} chars")
            
            text2 = st.text_area("Or paste", value=text2, height=200, key="t2")
        
        if st.button("🔍 Check", type="primary"):
            if not text1:
                st.error("Need submission text")
            else:
                with st.spinner("Analyzing..."):
                    paras = [p.strip() for p in text1.split('\n\n') if len(p.strip()) > 100]
                    
                    if len(paras) < 2:
                        sents = sent_tokenize(text1)
                        mid = len(sents) // 2
                        refs = [' '.join(sents[:mid]) if mid > 0 else text1[:len(text1)//2],
                                ' '.join(sents[mid:]) if mid > 0 else text1[len(text1)//2:]]
                    else:
                        refs = paras[:2]
                    
                    results = system.analyze_plagiarism(text1, refs, text2 if text2 else None)
                    
                    orig = 100.0
                    if 'intrinsic_detection' in results and results['intrinsic_detection']:
                        cons = results['intrinsic_detection']['avg_style_consistency']
                        if cons < 0.6:
                            orig -= 30
                        elif cons < 0.7:
                            orig -= 15
                    
                    nov = results['novelty_detection']['novelty_ratio']
                    orig = orig * nov
                    
                    if text2 and 'consistency_with_history' in results['stylometric_analysis']:
                        style_cons = results['stylometric_analysis']['consistency_with_history']
                        if style_cons < 0.5:
                            orig -= 20
                        elif style_cons < 0.7:
                            orig -= 10
                    
                    orig = max(0, min(100, orig))
                    
                    st.success("✅ Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        color = "🟢" if orig > 75 else "🟡" if orig > 50 else "🔴"
                        st.metric("Originality", f"{color} {orig:.1f}%")
                    with col2:
                        st.metric("Similarity", f"{100-orig:.1f}%")
                    with col3:
                        risk = "Low" if orig > 75 else "Medium" if orig > 50 else "High"
                        st.metric("Risk", risk)
                    
                    st.divider()
                    create_visualization(results)
    
    with tab2:
        st.header("Compare Assignments")
        st.info("📌 Upload multiple in any format")
        
        n = st.number_input("Number", 2, 20, 3)
        
        assigns = {}
        cols = st.columns(2)
        
        for i in range(n):
            with cols[i % 2]:
                st.subheader(f"Student {i+1}")
                name = st.text_input("Name", f"Student_{i+1}", key=f"n{i}")
                file = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key=f"cf{i}")
                
                text = ""
                if file:
                    with st.spinner("Extracting..."):
                        text = system.extract_text_from_file(file)
                    if text:
                        st.success(f"✅ {len(text)} chars")
                
                text = st.text_area("Or paste", value=text, height=150, key=f"ct{i}")
                
                if text:
                    assigns[name] = text
        
        if st.button("🔍 Compare", type="primary"):
            if len(assigns) < 2:
                st.error("Need 2+ assignments")
            else:
                with st.spinner("Comparing..."):
                    matrix, details, names = system.compare_multiple_assignments(assigns)
                    
                    st.success("✅ Complete!")
                    
                    st.subheader("📊 Matrix")
                    df = pd.DataFrame(matrix, columns=names, index=names)
                    st.dataframe(df.style.background_gradient(cmap='Reds', vmin=0, vmax=1), use_container_width=True)
                    
                    st.divider()
                    st.subheader("⚠️ Suspicious (>50%)")
                    
                    sus = []
                    for k, v in details.items():
                        if v['combined_score'] > 0.5:
                            sus.append({
                                'Pair': k,
                                'Score': f"{v['combined_score']:.1%}",
                                'Semantic': f"{v['semantic_similarity']:.1%}",
                                'Fingerprint': f"{v['fingerprint_similarity']:.1%}"
                            })
                    
                    if sus:
                        st.dataframe(pd.DataFrame(sus), use_container_width=True)
                        st.warning(f"🚨 {len(sus)} suspicious pairs!")
                    else:
                        st.success("✅ All unique")
    
    with tab3:
        st.header("About")
        st.markdown("""
        ### Features
        - 80+ languages (Hindi, Arabic, Chinese, etc.)
        - OCR for handwritten/printed text
        - PDF, DOCX, images supported
        - No references needed for originality check
        - Compare multiple assignments for collusion
        
        ### Detection Methods
        1. Semantic similarity (meaning)
        2. Document fingerprinting (winnowing)
        3. Stylometric analysis (writing style)
        4. Novelty detection (sentence-level)
        5. Intrinsic detection (internal consistency)
        """)

if __name__ == "__main__":
    main()
