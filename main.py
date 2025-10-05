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
    """Main plagiarism detection system"""
    
    def __init__(self, performance_mode="balanced"):
        self.embedding_models = {
            "fast": "paraphrase-multilingual-MiniLM-L12-v2",
            "balanced": "paraphrase-multilingual-mpnet-base-v2",
            "best": "sentence-transformers/all-mpnet-base-v2"
        }
        
        self.cross_encoders = {
            "fast": "cross-encoder/stsb-roberta-base",
            "best": "cross-encoder/stsb-roberta-large"
        }
        
        with st.spinner("🔄 Loading AI models..."):
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
        if self.ocr_reader is None:
            with st.spinner("🔄 Loading OCR model..."):
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        return self.ocr_reader
    
    def extract_text_from_image(self, image):
        reader = self.load_ocr()
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=0)
        return '\n'.join(results)
    
    def extract_text_from_pdf(self, pdf_file):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def extract_text_from_docx(self, docx_file):
        """Fixed DOCX extraction"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(docx_file.getvalue())
                tmp_path = tmp_file.name
            
            # Read the document
            doc = Document(tmp_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return text.strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""
    
    def calculate_originality_score(self, submitted_text, reference_texts):
        """
        Calculate originality score similar to Turnitin
        Returns percentage of unique content
        """
        submitted_embedding = self.embedding_model.encode([submitted_text], show_progress_bar=False)[0]
        
        # Sentence-level matching
        submitted_sentences = sent_tokenize(submitted_text)
        submitted_sent_embeddings = self.embedding_model.encode(submitted_sentences, show_progress_bar=False)
        
        # Collect all reference sentences
        all_ref_sentences = []
        for ref_text in reference_texts:
            all_ref_sentences.extend(sent_tokenize(ref_text))
        
        if not all_ref_sentences:
            return 100.0, []  # 100% original if no references
        
        ref_embeddings = self.model.encode(all_ref_sentences, show_progress_bar=False)
        
        # Match each sentence
        matched_sentences = []
        for i, (sent, emb) in enumerate(zip(submitted_sentences, submitted_sent_embeddings)):
            similarities = cosine_similarity([emb], ref_embeddings)[0]
            max_sim = np.max(similarities)
            max_idx = np.argmax(similarities)
            
            if max_sim > 0.75:  # Threshold for plagiarism
                matched_sentences.append({
                    'sentence': sent,
                    'similarity': float(max_sim),
                    'matched_with': all_ref_sentences[max_idx]
                })
        
        # Calculate originality percentage
        originality = ((len(submitted_sentences) - len(matched_sentences)) / len(submitted_sentences)) * 100
        
        return originality, matched_sentences
    
    def compare_multiple_assignments(self, assignments_dict):
        """
        Compare multiple student assignments against each other
        Returns similarity matrix and detailed comparisons
        """
        student_names = list(assignments_dict.keys())
        n_students = len(student_names)
        
        # Create similarity matrix
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
                
                # Store detailed comparison
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
            cross_similarity = self.cross_encoder.predict([(submitted_text[:512], ref_text[:512])])[0]
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
        st.subheader("🔍 Document Fingerprint Matching (Novel)")
        
        fp_data = pd.DataFrame({
            'Reference': list(results['fingerprint_matching'].keys()),
            'Fingerprint Similarity': list(results['fingerprint_matching'].values())
        })
        
        st.bar_chart(fp_data.set_index('Reference'))
    
    if results['novelty_detection']:
        st.subheader("✨ Novelty Detection Analysis (Novel)")
        
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
        st.subheader("📝 Writing Style Analysis (Novel)")
        
        if 'features' in results['stylometric_analysis']:
            features_df = pd.DataFrame([results['stylometric_analysis']['features']])
            st.dataframe(features_df, use_container_width=True)
        
        if 'consistency_with_history' in results['stylometric_analysis']:
            consistency = results['stylometric_analysis']['consistency_with_history']
            st.metric(
                "Style Consistency with Previous Work", 
                f"{consistency:.2%}",
                delta="Consistent" if consistency > 0.7 else "Inconsistent"
            )
    
    if results['intrinsic_detection']:
        st.subheader("🔎 Intrinsic Plagiarism Detection (Novel)")
        
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
    st.markdown("### Multi-Mode Detection: Originality Check & Assignment Comparison")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        performance_mode = st.selectbox(
            "Performance Mode",
            ["fast", "balanced", "best"],
            index=1
        )
        
        st.divider()
        
        st.header("📚 Detection Modes")
        st.markdown("""
        **Mode 1: Originality Check**
        - Check single document
        - Like Turnitin
        - Against reference sources
        
        **Mode 2: Assignment Comparison**
        - Compare multiple submissions
        - Detect student collusion
        - Cross-check all assignments
        """)
    
    # Load system
    system = load_system(performance_mode)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Originality Check (Turnitin Mode)", 
        "🔄 Compare Assignments (Collusion Detection)",
        "📷 OCR Input", 
        "ℹ️ About"
    ])
    
    # TAB 1: ORIGINALITY CHECK MODE
    with tab1:
        st.header("Single Document Originality Check")
        st.markdown("*Check one assignment against reference sources for plagiarism*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Student Submission")
            
            uploaded_file = st.file_uploader(
                "Upload Assignment (PDF/DOCX)",
                type=['pdf', 'docx'],
                key="orig_student_file",
                help="Upload PDF or Word document"
            )
            
            submitted_text = ""
            
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    submitted_text = system.extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    submitted_text = system.extract_text_from_docx(uploaded_file)
                
                if submitted_text:
                    st.success(f"✅ Extracted {len(submitted_text)} characters")
                else:
                    st.error("❌ No text extracted. Check if document has readable text.")
            
            submitted_text = st.text_area(
                "Or paste text directly",
                value=submitted_text,
                height=300,
                key="orig_text_area",
                placeholder="Enter the student's assignment text here..."
            )
            
            st.subheader("Student's Previous Work (Optional)")
            student_history = st.text_area(
                "For style comparison",
                height=150,
                key="orig_history",
                placeholder="Previous assignment for style verification..."
            )
        
        with col2:
            st.subheader("Reference Sources")
            num_references = st.number_input("Number of reference documents", 1, 5, 1, key="orig_num_ref")
            
            reference_texts = []
            for i in range(num_references):
                ref_file = st.file_uploader(
                    f"Upload Reference {i+1} (PDF/DOCX)",
                    type=['pdf', 'docx'],
                    key=f"orig_ref_file_{i}"
                )
                
                ref_text = ""
                if ref_file:
                    if ref_file.type == "application/pdf":
                        ref_text = system.extract_text_from_pdf(ref_file)
                    elif ref_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        ref_text = system.extract_text_from_docx(ref_file)
                
                ref_text = st.text_area(
                    f"Reference {i+1}",
                    value=ref_text,
                    height=150,
                    key=f"orig_ref_{i}",
                    placeholder=f"Paste or upload reference {i+1}..."
                )
                if ref_text:
                    reference_texts.append(ref_text)
        
        if st.button("🔍 Check Originality", type="primary", key="orig_analyze"):
            if not submitted_text:
                st.error("Please provide the submitted assignment.")
            elif not reference_texts:
                st.error("Please provide at least one reference document.")
            else:
                with st.spinner("🔄 Analyzing originality..."):
                    results = system.analyze_plagiarism(
                        submitted_text,
                        reference_texts,
                        student_history if student_history else None
                    )
                    
                    st.success("✅ Originality Check Complete!")
                    
                    # Display originality score prominently
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            "Originality Score",
                            f"{results['originality_percentage']:.1f}%",
                            help="Percentage of original content"
                        )
                    with col2:
                        st.metric(
                            "Plagiarism Score",
                            f"{results['overall_score']:.1%}"
                        )
                    with col3:
                        st.metric(
                            "Risk Level",
                            results['risk_level']
                        )
                    
                    st.divider()
                    
                    create_visualization(results)
                    
                    with st.expander("📥 Export Report"):
                        st.json(results)
    
    # TAB 2: ASSIGNMENT COMPARISON MODE
    with tab2:
        st.header("Compare Multiple Assignments (Collusion Detection)")
        st.markdown("*Upload multiple student assignments to check for copying between students*")
        
        st.info("📌 This mode compares all assignments against each other to detect potential collusion/copying")
        
        num_assignments = st.number_input(
            "Number of assignments to compare",
            min_value=2,
            max_value=20,
            value=3,
            key="comp_num"
        )
        
        assignments = {}
        
        cols = st.columns(2)
        
        for i in range(num_assignments):
            col_idx = i % 2
            with cols[col_idx]:
                st.subheader(f"Student {i+1}")
                
                student_name = st.text_input(
                    f"Student {i+1} Name",
                    value=f"Student_{i+1}",
                    key=f"comp_name_{i}"
                )
                
                uploaded_file = st.file_uploader(
                    f"Upload Assignment",
                    type=['pdf', 'docx'],
                    key=f"comp_file_{i}"
                )
                
                assignment_text = ""
                
                if uploaded_file:
                    if uploaded_file.type == "application/pdf":
                        assignment_text = system.extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        assignment_text = system.extract_text_from_docx(uploaded_file)
                    
                    if assignment_text:
                        st.success(f"✅ {len(assignment_text)} characters")
                
                assignment_text = st.text_area(
                    f"Or paste text",
                    value=assignment_text,
                    height=150,
                    key=f"comp_text_{i}"
                )
                
                if assignment_text:
                    assignments[student_name] = assignment_text
        
        if st.button("🔍 Compare All Assignments", type="primary", key="comp_analyze"):
            if len(assignments) < 2:
                st.error("Please provide at least 2 assignments to compare.")
            else:
                with st.spinner("🔄 Comparing assignments..."):
                    similarity_matrix, detailed_comparisons, student_names = system.compare_multiple_assignments(assignments)
                    
                    st.success("✅ Comparison Complete!")
                    
                    # Display similarity matrix
                    st.subheader("📊 Similarity Matrix")
                    st.markdown("*Higher scores indicate potential copying/collusion*")
                    
                    df_matrix = pd.DataFrame(
                        similarity_matrix,
                        columns=student_names,
                        index=student_names
                    )
                    
                    # Color code the matrix
                    st.dataframe(
                        df_matrix.style.background_gradient(cmap='Reds', vmin=0, vmax=1),
                        use_container_width=True
                    )
                    
                    st.divider()
                    
                    # Show suspicious pairs
                    st.subheader("⚠️ Suspicious Pairs (Similarity > 50%)")
                    
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
                        df_suspicious = pd.DataFrame(suspicious_pairs)
                        st.dataframe(df_suspicious, use_container_width=True)
                        
                        st.warning(f"🚨 Found {len(suspicious_pairs)} suspicious pair(s) with high similarity!")
                    else:
                        st.success("✅ No suspicious similarities detected. All assignments appear unique.")
                    
                    st.divider()
                    
                    # Detailed comparisons
                    with st.expander("📋 View All Pairwise Comparisons"):
                        for pair_key, comparison in detailed_comparisons.items():
                            st.markdown(f"**{pair_key}**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Combined Score", f"{comparison['combined_score']:.1%}")
                            with col2:
                                st.metric("Semantic", f"{comparison['semantic_similarity']:.1%}")
                            with col3:
                                st.metric("Fingerprint", f"{comparison['fingerprint_similarity']:.1%}")
                            st.divider()
                    
                    # Export results
                    with st.expander("📥 Export Comparison Report"):
                        export_data = {
                            'similarity_matrix': df_matrix.to_dict(),
                            'detailed_comparisons': detailed_comparisons,
                            'suspicious_pairs': suspicious_pairs
                        }
                        st.json(export_data)
    
    # TAB 3: OCR INPUT
    with tab3:
        st.header("OCR Text Extraction")
        st.markdown("Upload an image of handwritten or printed assignment")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            key="ocr_image",
            help="Upload a clear image of the document"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("🔍 Extract Text", key="ocr_extract"):
                    with st.spinner("🔄 Extracting text..."):
                        extracted_text = system.extract_text_from_image(image)
                        
                        st.success("✅ Text extracted!")
                        st.text_area(
                            "Extracted Text",
                            extracted_text,
                            height=300
                        )
                        
                        st.info("💡 Copy this text and use it in either detection mode")
    
    # TAB 4: ABOUT
    with tab4:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Two Detection Modes
        
        #### Mode 1: Originality Check (Like Turnitin)
        - Check a **single assignment** against reference sources
        - Detects plagiarism from external sources
        - Provides originality percentage
        - Identifies specific copied passages
        - Includes writing style analysis
        
        #### Mode 2: Assignment Comparison (Collusion Detection)
        - Compare **multiple student assignments** against each other
        - Detects copying between students
        - Creates similarity matrix for all submissions
        - Identifies suspicious pairs automatically
        - Useful for detecting collaboration/collusion
        
        ### 🔬 Novel Features
        
        **1. Stylometric Fingerprinting**  
        Analyzes unique writing patterns to verify authorship
        
        **2. Winnowing Algorithm**  
        Efficient document fingerprinting for precise matching
        
        **3. Sentence-Level Novelty Detection**  
        Identifies exactly which sentences are plagiarized
        
        **4. Intrinsic Plagiarism Detection**  
        Detects style inconsistencies within a single document
        
        **5. Multi-Format Support**  
        PDF, DOCX, and OCR for images
        
        ### 📊 Similarity Scoring
        
        - **Semantic Similarity**: Meaning-based comparison using transformer models
        - **Fingerprint Matching**: Structural pattern matching using hashing
        - **Combined Score**: Weighted average for final plagiarism score
        
        ### 🚀 Why This is Better
        
        Unlike simple text matching tools, this system combines:
        - **Semantic** understanding (what text means)
        - **Syntactic** analysis (how it's structured)
        - **Stylometric** profiling (writing style DNA)
        - **Multi-document** comparison (collusion detection)
        
        ### 📈 Interpretation Guide
        
        **Originality Check:**
        - **>90%**: Highly original
        - **70-90%**: Good with proper citations
        - **50-70%**: Moderate concern
        - **<50%**: High plagiarism risk
        
        **Assignment Comparison:**
        - **>70%**: Likely copying/collusion
        - **50-70%**: Suspicious similarity
        - **30-50%**: Some shared content
        - **<30%**: Normal variation
        """)

if __name__ == "__main__":
    main()
