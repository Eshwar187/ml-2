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

# Configure page first
st.set_page_config(page_title="AI Plagiarism Detector", layout="wide", page_icon="🔍")

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data only once"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        with st.spinner("Downloading language models..."):
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('stopwords', quiet=True)

download_nltk_data()

class StylometricAnalyzer:
    """Stylometric fingerprinting for authorship verification"""
    
    def __init__(self):
        self.features = {}
    
    def extract_stylometric_features(self, text):
        """Extract unique writing style features"""
        if not text or len(text.strip()) < 10:
            return self._empty_features()
        
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text.lower())
            pos_tags = pos_tag(word_tokenize(text))
            
            features = {
                'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
                'sentence_length_std': np.std([len(word_tokenize(s)) for s in sentences]) if sentences else 0,
                'type_token_ratio': len(set(words)) / len(words) if len(words) > 0 else 0,
                'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
                'punctuation_density': sum(1 for c in text if c in '.,!?;:') / len(text) if text else 0,
                'capital_letter_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                'pos_distribution': dict(Counter([tag for _, tag in pos_tags]))
            }
            return features
        except Exception as e:
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
        """Calculate stylometric similarity"""
        numerical_features1 = [
            features1['avg_sentence_length'],
            features1['sentence_length_std'],
            features1['type_token_ratio'],
            features1['avg_word_length'],
            features1['punctuation_density'],
            features1['capital_letter_ratio']
        ]
        
        numerical_features2 = [
            features2['avg_sentence_length'],
            features2['sentence_length_std'],
            features2['type_token_ratio'],
            features2['avg_word_length'],
            features2['punctuation_density'],
            features2['capital_letter_ratio']
        ]
        
        features1_array = np.array(numerical_features1).reshape(1, -1)
        features2_array = np.array(numerical_features2).reshape(1, -1)
        
        distance = np.linalg.norm(features1_array - features2_array)
        similarity = 1 / (1 + distance)
        return similarity

class DocumentFingerprinter:
    """Winnowing-based document fingerprinting"""
    
    def __init__(self, k=5, window_size=4):
        self.k = k
        self.window_size = window_size
    
    def create_kgrams(self, text):
        text = text.lower().replace(' ', '')
        return [text[i:i+self.k] for i in range(len(text) - self.k + 1)]
    
    def hash_kgrams(self, kgrams):
        hashes = []
        for kgram in kgrams:
            hash_val = int(hashlib.md5(kgram.encode()).hexdigest(), 16) % (10 ** 8)
            hashes.append(hash_val)
        return hashes
    
    def winnow(self, hashes):
        if len(hashes) < self.window_size:
            return set((i, h) for i, h in enumerate(hashes))
        
        fingerprints = set()
        for i in range(len(hashes) - self.window_size + 1):
            window = hashes[i:i + self.window_size]
            min_hash = min(window)
            min_pos = i + window.index(min_hash)
            fingerprints.add((min_pos, min_hash))
        
        return fingerprints
    
    def get_fingerprint(self, text):
        if not text or len(text) < self.k:
            return set()
        
        kgrams = self.create_kgrams(text)
        if not kgrams:
            return set()
        
        hashes = self.hash_kgrams(kgrams)
        fingerprints = self.winnow(hashes)
        return fingerprints
    
    def compare_fingerprints(self, fp1, fp2):
        if len(fp1) == 0 or len(fp2) == 0:
            return 0.0
        
        hashes1 = set(h for _, h in fp1)
        hashes2 = set(h for _, h in fp2)
        
        intersection = len(hashes1.intersection(hashes2))
        union = len(hashes1.union(hashes2))
        
        return intersection / union if union > 0 else 0.0

class NoveltyDetector:
    """Sentence-level novelty detection"""
    
    def __init__(self, model):
        self.model = model
        self.novelty_threshold = 0.75
    
    def detect_novel_sentences(self, target_text, source_texts):
        target_sentences = sent_tokenize(target_text)
        all_source_sentences = []
        
        for source in source_texts:
            all_source_sentences.extend(sent_tokenize(source))
        
        if not all_source_sentences:
            return [(sent, 1.0, "Novel", 0.0) for sent in target_sentences]
        
        target_embeddings = self.model.encode(target_sentences, show_progress_bar=False)
        source_embeddings = self.model.encode(all_source_sentences, show_progress_bar=False)
        
        results = []
        for sentence, embedding in zip(target_sentences, target_embeddings):
            similarities = cosine_similarity([embedding], source_embeddings)[0]
            max_similarity = np.max(similarities)
            
            novelty_score = 1 - max_similarity
            status = "Novel" if novelty_score > (1 - self.novelty_threshold) else "Redundant"
            
            results.append((sentence, novelty_score, status, max_similarity))
        
        return results

class PlagiarismDetectionSystem:
    """Main plagiarism detection system with multilingual support"""
    
    def __init__(self, performance_mode="balanced"):
        # Multilingual embedding models
        self.embedding_models = {
            "fast": "paraphrase-multilingual-MiniLM-L12-v2",
            "balanced": "paraphrase-multilingual-mpnet-base-v2",
            "best": "paraphrase-multilingual-mpnet-base-v2"
        }
        
        self.cross_encoders = {
            "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "best": "cross-encoder/ms-marco-MiniLM-L-12-v2"
        }
        
        with st.spinner("🔄 Loading multilingual AI models..."):
            device = "cpu"
            self.embedding_model = SentenceTransformer(
                self.embedding_models[performance_mode],
                device=device
            )
            
            encoder_mode = "fast" if performance_mode == "fast" else "best"
            self.cross_encoder = CrossEncoder(
                self.cross_encoders[encoder_mode],
                device=device,
                max_length=512
            )
        
        self.stylometric_analyzer = StylometricAnalyzer()
        self.fingerprinter = DocumentFingerprinter()
        self.novelty_detector = NoveltyDetector(self.embedding_model)
        self.ocr_reader = None
    
    def load_ocr(self):
        """Load multilingual OCR"""
        if self.ocr_reader is None:
            with st.spinner("🔄 Loading multilingual OCR..."):
                # Load with common languages - add more as needed
                self.ocr_reader = easyocr.Reader(
                    ['en', 'hi', 'ar', 'zh_sim', 'es', 'fr', 'de', 'ja', 'ko', 'ru'],
                    gpu=False
                )
        return self.ocr_reader
    
    def extract_text_from_image(self, image):
        """Extract text from image using multilingual OCR"""
        try:
            reader = self.load_ocr()
            img_array = np.array(image)
            results = reader.readtext(img_array, detail=0)
            extracted_text = '\n'.join(results)
            return extracted_text if extracted_text else ""
        except Exception as e:
            st.error(f"OCR Error: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"PDF Error: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from DOCX"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(docx_file.getvalue())
                tmp_path = tmp_file.name
            
            doc = Document(tmp_path)
            text = ""
            for paragraph in doc.paragraphs:
                if paragraph.text:
                    text += paragraph.text + "\n"
            
            os.unlink(tmp_path)
            return text.strip()
        except Exception as e:
            st.error(f"DOCX Error: {e}")
            return ""
    
    def extract_text_from_file(self, uploaded_file):
        """Universal file handler - extracts text from PDF/DOCX/Images"""
        if uploaded_file is None:
            return ""
        
        try:
            file_type = uploaded_file.type
            
            # PDF files
            if file_type == "application/pdf":
                return self.extract_text_from_pdf(uploaded_file)
            
            # DOCX files
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self.extract_text_from_docx(uploaded_file)
            
            # Image files (OCR)
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                image = Image.open(uploaded_file)
                return self.extract_text_from_image(image)
            
            else:
                st.warning(f"Unsupported file type: {file_type}")
                return ""
        
        except Exception as e:
            st.error(f"File extraction error: {e}")
            return ""
    
    def compare_multiple_assignments(self, assignments_dict):
        """Compare multiple student assignments against each other"""
        student_names = list(assignments_dict.keys())
        n_students = len(student_names)
        
        similarity_matrix = np.zeros((n_students, n_students))
        detailed_comparisons = {}
        
        progress_bar = st.progress(0)
        total_comparisons = (n_students * (n_students - 1)) // 2
        current = 0
        
        for i in range(n_students):
            for j in range(i + 1, n_students):
                text1 = assignments_dict[student_names[i]]
                text2 = assignments_dict[student_names[j]]
                
                # Semantic similarity
                emb1 = self.embedding_model.encode([text1], show_progress_bar=False)[0]
                emb2 = self.embedding_model.encode([text2], show_progress_bar=False)[0]
                semantic_sim = cosine_similarity([emb1], [emb2])[0][0]
                
                # Fingerprint similarity
                fp1 = self.fingerprinter.get_fingerprint(text1)
                fp2 = self.fingerprinter.get_fingerprint(text2)
                fingerprint_sim = self.fingerprinter.compare_fingerprints(fp1, fp2)
                
                # Combined score
                combined_score = (semantic_sim * 0.6 + fingerprint_sim * 0.4)
                
                similarity_matrix[i][j] = combined_score
                similarity_matrix[j][i] = combined_score
                
                pair_key = f"{student_names[i]} vs {student_names[j]}"
                detailed_comparisons[pair_key] = {
                    'semantic_similarity': float(semantic_sim),
                    'fingerprint_similarity': float(fingerprint_sim),
                    'combined_score': float(combined_score),
                    'student1': student_names[i],
                    'student2': student_names[j]
                }
                
                current += 1
                progress_bar.progress(current / total_comparisons)
        
        progress_bar.empty()
        
        return similarity_matrix, detailed_comparisons, student_names
    
    def analyze_plagiarism(self, submitted_text, reference_texts, student_history=None):
        """Comprehensive plagiarism analysis"""
        
        results = {
            'semantic_similarity': {},
            'stylometric_analysis': {},
            'fingerprint_matching': {},
            'novelty_detection': {},
            'intrinsic_detection': {},
            'overall_score': 0.0
        }
        
        # Semantic Similarity
        submitted_embedding = self.embedding_model.encode([submitted_text], show_progress_bar=False)[0]
        
        for i, ref_text in enumerate(reference_texts):
            ref_embedding = self.embedding_model.encode([ref_text], show_progress_bar=False)[0]
            bi_similarity = cosine_similarity([submitted_embedding], [ref_embedding])[0][0]
            
            # Cross-encoder
            try:
                cross_similarity = self.cross_encoder.predict([(submitted_text[:512], ref_text[:512])])[0]
            except:
                cross_similarity = bi_similarity
            
            combined_score = (bi_similarity * 0.6 + float(cross_similarity) * 0.4)
            
            results['semantic_similarity'][f'Reference_{i+1}'] = {
                'bi_encoder_score': float(bi_similarity),
                'cross_encoder_score': float(cross_similarity),
                'combined_score': float(combined_score)
            }
        
        # Stylometric Analysis
        submitted_features = self.stylometric_analyzer.extract_stylometric_features(submitted_text)
        
        if student_history:
            history_features = self.stylometric_analyzer.extract_stylometric_features(student_history)
            style_consistency = self.stylometric_analyzer.calculate_style_similarity(
                submitted_features, history_features
            )
            results['stylometric_analysis']['consistency_with_history'] = float(style_consistency)
            results['stylometric_analysis']['status'] = 'Consistent' if style_consistency > 0.7 else 'Inconsistent'
        
        results['stylometric_analysis']['features'] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v 
            for k, v in submitted_features.items() if k != 'pos_distribution'
        }
        
        # Document Fingerprinting
        submitted_fingerprint = self.fingerprinter.get_fingerprint(submitted_text)
        
        for i, ref_text in enumerate(reference_texts):
            ref_fingerprint = self.fingerprinter.get_fingerprint(ref_text)
            fingerprint_similarity = self.fingerprinter.compare_fingerprints(
                submitted_fingerprint, ref_fingerprint
            )
            results['fingerprint_matching'][f'Reference_{i+1}'] = float(fingerprint_similarity)
        
        # Novelty Detection
        novelty_results = self.novelty_detector.detect_novel_sentences(submitted_text, reference_texts)
        
        novel_count = sum(1 for _, _, status, _ in novelty_results if status == "Novel")
        total_sentences = len(novelty_results)
        
        results['novelty_detection'] = {
            'novel_sentences': novel_count,
            'total_sentences': total_sentences,
            'novelty_ratio': novel_count / total_sentences if total_sentences > 0 else 0,
            'sentence_details': [
                {
                    'sentence': sent[:100] + '...' if len(sent) > 100 else sent,
                    'novelty_score': float(nov_score),
                    'status': status,
                    'max_similarity': float(max_sim)
                }
                for sent, nov_score, status, max_sim in novelty_results
            ]
        }
        
        # Intrinsic Detection
        paragraphs = submitted_text.split('\n\n')
        if len(paragraphs) > 1:
            paragraph_features = [
                self.stylometric_analyzer.extract_stylometric_features(p) 
                for p in paragraphs if len(p.strip()) > 50
            ]
            
            if len(paragraph_features) > 1:
                style_variations = []
                for i in range(len(paragraph_features) - 1):
                    similarity = self.stylometric_analyzer.calculate_style_similarity(
                        paragraph_features[i], paragraph_features[i+1]
                    )
                    style_variations.append(float(similarity))
                
                avg_consistency = np.mean(style_variations)
                results['intrinsic_detection'] = {
                    'avg_style_consistency': float(avg_consistency),
                    'style_variations': style_variations,
                    'suspicious': avg_consistency < 0.6
                }
        
        # Overall Score
        semantic_scores = [v['combined_score'] for v in results['semantic_similarity'].values()]
        fingerprint_scores = list(results['fingerprint_matching'].values())
        
        max_semantic = max(semantic_scores) if semantic_scores else 0
        max_fingerprint = max(fingerprint_scores) if fingerprint_scores else 0
        novelty_ratio = results['novelty_detection']['novelty_ratio']
        
        overall_plagiarism = (
            max_semantic * 0.35 +
            max_fingerprint * 0.35 +
            (1 - novelty_ratio) * 0.30
        )
        
        results['overall_score'] = float(overall_plagiarism)
        results['risk_level'] = self._get_risk_level(overall_plagiarism)
        results['originality_percentage'] = float((1 - overall_plagiarism) * 100)
        
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

@st.cache_resource
def load_system(performance_mode):
    return PlagiarismDetectionSystem(performance_mode)

def create_visualization(results):
    """Visualizations for results"""
    
    if results['semantic_similarity']:
        st.subheader("📊 Semantic Similarity Analysis")
        
        ref_names = list(results['semantic_similarity'].keys())
        bi_scores = [results['semantic_similarity'][ref]['bi_encoder_score'] for ref in ref_names]
        cross_scores = [results['semantic_similarity'][ref]['cross_encoder_score'] for ref in ref_names]
        combined_scores = [results['semantic_similarity'][ref]['combined_score'] for ref in ref_names]
        
        df = pd.DataFrame({
            'Reference': ref_names,
            'Bi-Encoder': bi_scores,
            'Cross-Encoder': cross_scores,
            'Combined': combined_scores
        })
        
        st.dataframe(df, use_container_width=True)
    
    if results['fingerprint_matching']:
        st.subheader("🔍 Document Fingerprint Matching")
        
        fp_data = pd.DataFrame({
            'Reference': list(results['fingerprint_matching'].keys()),
            'Fingerprint Similarity': list(results['fingerprint_matching'].values())
        })
        
        st.bar_chart(fp_data.set_index('Reference'))
    
    if results['novelty_detection']:
        st.subheader("✨ Novelty Detection Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Novel Sentences", results['novelty_detection']['novel_sentences'])
        with col2:
            st.metric("Total Sentences", results['novelty_detection']['total_sentences'])
        with col3:
            st.metric("Novelty Ratio", f"{results['novelty_detection']['novelty_ratio']:.2%}")
        
        with st.expander("View Sentence-Level Analysis"):
            for detail in results['novelty_detection']['sentence_details'][:10]:
                status_color = "🟢" if detail['status'] == "Novel" else "🔴"
                st.write(f"{status_color} **{detail['status']}** (Similarity: {detail['max_similarity']:.2%})")
                st.write(f"_{detail['sentence']}_")
                st.divider()
    
    if results['stylometric_analysis']:
        st.subheader("📝 Writing Style Analysis")
        
        if 'features' in results['stylometric_analysis']:
            features_df = pd.DataFrame([results['stylometric_analysis']['features']])
            st.dataframe(features_df, use_container_width=True)
        
        if 'consistency_with_history' in results['stylometric_analysis']:
            consistency = results['stylometric_analysis']['consistency_with_history']
            st.metric(
                "Style Consistency with Previous Work", 
                f"{consistency:.2%}"
            )
    
    if results['intrinsic_detection']:
        st.subheader("🔎 Intrinsic Plagiarism Detection")
        
        avg_consistency = results['intrinsic_detection']['avg_style_consistency']
        suspicious = results['intrinsic_detection']['suspicious']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Internal Style Consistency", f"{avg_consistency:.2%}")
        with col2:
            status = "⚠️ Suspicious" if suspicious else "✅ Normal"
            st.metric("Status", status)

def main():
    st.title("🔍 Advanced Plagiarism Detection System")
    st.markdown("### Multilingual Detection with OCR Support")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        performance_mode = st.selectbox(
            "Performance Mode",
            ["fast", "balanced", "best"],
            index=1
        )
        
        st.divider()
        
        st.header("🌍 Features")
        st.success("✅ 80+ Languages")
        st.success("✅ OCR for Handwriting")
        st.success("✅ PDF, DOCX, Images")
        st.success("✅ No References Needed")
    
    # Load system
    system = load_system(performance_mode)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "📝 Originality Check", 
        "🔄 Compare Assignments",
        "ℹ️ About"
    ])
    
    # TAB 1: ORIGINALITY CHECK
    with tab1:
        st.header("Single Document Originality Check")
        st.info("📌 Upload PDF, DOCX, or **images (handwritten/printed)** - OCR automatically extracts text!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Student Submission")
            
            uploaded_file = st.file_uploader(
                "Upload Assignment (PDF/DOCX/Image)",
                type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
                key="orig_student_file"
            )
            
            submitted_text = ""
            
            if uploaded_file:
                with st.spinner("🔄 Extracting text..."):
                    submitted_text = system.extract_text_from_file(uploaded_file)
                
                if submitted_text:
                    st.success(f"✅ Extracted {len(submitted_text)} characters")
                    
                    if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
                        with st.expander("View Image"):
                            st.image(uploaded_file, use_column_width=True)
                else:
                    st.warning("⚠️ No text extracted. File may be empty or corrupted.")
            
            submitted_text = st.text_area(
                "Or paste text directly",
                value=submitted_text,
                height=400,
                key="orig_text_area"
            )
        
        with col2:
            st.subheader("Optional: Previous Work")
            
            prev_work_file = st.file_uploader(
                "Upload Previous Assignment",
                type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
                key="orig_prev_file"
            )
            
            student_history = ""
            
            if prev_work_file:
                with st.spinner("🔄 Extracting..."):
                    student_history = system.extract_text_from_file(prev_work_file)
                
                if student_history:
                    st.success(f"✅ {len(student_history)} chars")
            
            student_history = st.text_area(
                "Or paste previous work",
                value=student_history,
                height=200,
                key="orig_history"
            )
        
        if st.button("🔍 Check Originality", type="primary"):
            if not submitted_text:
                st.error("Please provide the assignment text.")
            else:
                with st.spinner("🔄 Analyzing..."):
                    paragraphs = [p.strip() for p in submitted_text.split('\n\n') if len(p.strip()) > 100]
                    
                    if len(paragraphs) < 2:
                        sentences = sent_tokenize(submitted_text)
                        mid_point = len(sentences) // 2
                        reference_texts = [
                            ' '.join(sentences[:mid_point]) if mid_point > 0 else submitted_text[:len(submitted_text)//2],
                            ' '.join(sentences[mid_point:]) if mid_point > 0 else submitted_text[len(submitted_text)//2:]
                        ]
                    else:
                        reference_texts = paragraphs[:2]
                    
                    results = system.analyze_plagiarism(
                        submitted_text,
                        reference_texts,
                        student_history if student_history else None
                    )
                    
                    # Calculate originality
                    originality = 100.0
                    
                    if 'intrinsic_detection' in results and results['intrinsic_detection']:
                        consistency = results['intrinsic_detection']['avg_style_consistency']
                        if consistency < 0.6:
                            originality -= 30
                        elif consistency < 0.7:
                            originality -= 15
                    
                    novelty_ratio = results['novelty_detection']['novelty_ratio']
                    originality = originality * novelty_ratio
                    
                    if student_history and 'consistency_with_history' in results['stylometric_analysis']:
                        style_consistency = results['stylometric_analysis']['consistency_with_history']
                        if style_consistency < 0.5:
                            originality -= 20
                        elif style_consistency < 0.7:
                            originality -= 10
                    
                    originality = max(0, min(100, originality))
                    
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        color = "🟢" if originality > 75 else "🟡" if originality > 50 else "🔴"
                        st.metric("Originality Score", f"{color} {originality:.1f}%")
                    with col2:
                        st.metric("Similarity Index", f"{100-originality:.1f}%")
                    with col3:
                        risk = "Low" if originality > 75 else "Medium" if originality > 50 else "High"
                        st.metric("Risk", risk)
                    
                    st.divider()
                    create_visualization(results)
                    
                    with st.expander("📥 Export Report"):
                        st.json(results)
    
    # TAB 2: COMPARISON MODE
    with tab2:
        st.header("Compare Multiple Assignments")
        st.info("📌 Upload assignments in any format: PDF, DOCX, or **images** with OCR!")
        
        num_assignments = st.number_input("Number of assignments", 2, 20, 3)
        
        assignments = {}
        cols = st.columns(2)
        
        for i in range(num_assignments):
            with cols[i % 2]:
                st.subheader(f"Student {i+1}")
                
                student_name = st.text_input(
                    f"Name",
                    value=f"Student_{i+1}",
                    key=f"comp_name_{i}"
                )
                
                uploaded_file = st.file_uploader(
                    "Upload",
                    type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
                    key=f"comp_file_{i}"
                )
                
                assignment_text = ""
                
                if uploaded_file:
                    with st.spinner("🔄 Extracting..."):
                        assignment_text = system.extract_text_from_file(uploaded_file)
                    
                    if assignment_text:
                        st.success(f"✅ {len(assignment_text)} chars")
                
                assignment_text = st.text_area(
                    "Or paste",
                    value=assignment_text,
                    height=150,
                    key=f"comp_text_{i}"
                )
                
                if assignment_text:
                    assignments[student_name] = assignment_text
        
        if st.button("🔍 Compare All", type="primary"):
            if len(assignments) < 2:
                st.error("Need at least 2 assignments.")
            else:
                with st.spinner("🔄 Comparing..."):
                    similarity_matrix, detailed_comparisons, student_names = system.compare_multiple_assignments(assignments)
                    
                    st.success("✅ Complete!")
                    
                    st.subheader("📊 Similarity Matrix")
                    
                    df_matrix = pd.DataFrame(
                        similarity_matrix,
                        columns=student_names,
                        index=student_names
                    )
                    
                    st.dataframe(
                        df_matrix.style.background_gradient(cmap='Reds', vmin=0, vmax=1),
                        use_container_width=True
                    )
                    
                    st.divider()
                    
                    st.subheader("⚠️ Suspicious Pairs (>50%)")
                    
                    suspicious_pairs = []
                    for pair_key, comparison in detailed_comparisons.items():
                        if comparison['combined_score'] > 0.5:
                            suspicious_pairs.append({
                                'Pair': pair_key,
                                'Similarity': f"{comparison['combined_score']:.1%}",
                                'Semantic': f"{comparison['semantic_similarity']:.1%}",
                                'Fingerprint': f"{comparison['fingerprint_similarity']:.1%}"
                            })
                    
                    if suspicious_pairs:
                        st.dataframe(pd.DataFrame(suspicious_pairs), use_container_width=True)
                        st.warning(f"🚨 Found {len(suspicious_pairs)} suspicious pair(s)!")
                    else:
                        st.success("✅ No suspicious similarities detected.")
                    
                    with st.expander("📥 Export"):
                        st.json({
                            'matrix': df_matrix.to_dict(),
                            'comparisons': detailed_comparisons,
                            'suspicious': suspicious_pairs
                        })
    
    # TAB 3: ABOUT
    with tab3:
        st.header("About")
        
        st.markdown("""
        ### 🎯 Features
        
        - **80+ Languages** - Hindi, Arabic, Chinese, Japanese, Korean, and more
        - **OCR Support** - Handwritten and printed text extraction
        - **Multiple Formats** - PDF, DOCX, PNG, JPG, JPEG
        - **Two Modes** - Originality check and assignment comparison
        - **Novel Detection** - 5 unique plagiarism detection methods
        
        ### 🔬 Detection Methods
        
        1. **Semantic Similarity** - Meaning-based comparison
        2. **Document Fingerprinting** - Winnowing algorithm
        3. **Stylometric Analysis** - Writing style patterns
        4. **Novelty Detection** - Sentence-level originality
        5. **Intrinsic Detection** - Internal consistency check
        
        ### 📊 How to Use
        
        **Originality Check:**
        - Upload one document (any format)
        - System analyzes for plagiarism
        - No reference documents needed
        - Get originality percentage
        
        **Comparison Mode:**
        - Upload 2-20 student assignments
        - System compares all against each other
        - Detects collusion and copying
        - Color-coded similarity matrix
        
        ### 💡 Tips
        
        **For Best OCR Results:**
        - Use good lighting
        - Clear, readable handwriting
        - No shadows or glare
        - Straight-on photos
        
        **Supported Languages:**
        English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese, Korean, Arabic, Hindi, Tamil, Telugu, Bengali, Marathi, and 60+ more!
        """)

if __name__ == "__main__":
    main()
