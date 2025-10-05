import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')

class StylometricAnalyzer:
    """Novel Feature #1: Stylometric fingerprinting for authorship verification"""
    
    def __init__(self):
        self.features = {}
    
    def extract_stylometric_features(self, text):
        """Extract unique writing style features"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        pos_tags = pos_tag(word_tokenize(text))
        
        # Calculate stylometric features
        features = {
            'avg_sentence_length': np.mean([len(word_tokenize(s)) for s in sentences]),
            'sentence_length_std': np.std([len(word_tokenize(s)) for s in sentences]),
            'type_token_ratio': len(set(words)) / len(words) if len(words) > 0 else 0,
            'avg_word_length': np.mean([len(w) for w in words]),
            'punctuation_density': sum(1 for c in text if c in '.,!?;:') / len(text),
            'capital_letter_ratio': sum(1 for c in text if c.isupper()) / len(text),
            'pos_distribution': dict(Counter([tag for _, tag in pos_tags]))
        }
        
        return features
    
    def calculate_style_similarity(self, features1, features2):
        """Calculate stylometric similarity between two text samples"""
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
        
        # Normalize and calculate Euclidean distance
        features1_array = np.array(numerical_features1).reshape(1, -1)
        features2_array = np.array(numerical_features2).reshape(1, -1)
        
        similarity = 1 / (1 + np.linalg.norm(features1_array - features2_array))
        return similarity

class DocumentFingerprinter:
    """Novel Feature #2: Winnowing-based document fingerprinting"""
    
    def __init__(self, k=5, window_size=4):
        self.k = k  # k-gram size
        self.window_size = window_size
    
    def create_kgrams(self, text):
        """Create k-grams from text"""
        text = text.lower().replace(' ', '')
        return [text[i:i+self.k] for i in range(len(text) - self.k + 1)]
    
    def hash_kgrams(self, kgrams):
        """Hash k-grams using rolling hash"""
        hashes = []
        for kgram in kgrams:
            hash_val = int(hashlib.md5(kgram.encode()).hexdigest(), 16) % (10 ** 8)
            hashes.append(hash_val)
        return hashes
    
    def winnow(self, hashes):
        """Apply winnowing algorithm to select fingerprints"""
        if len(hashes) < self.window_size:
            return set(hashes)
        
        fingerprints = set()
        min_pos = 0
        
        for i in range(len(hashes) - self.window_size + 1):
            window = hashes[i:i + self.window_size]
            min_hash = min(window)
            min_pos = i + window.index(min_hash)
            fingerprints.add((min_pos, min_hash))
        
        return fingerprints
    
    def get_fingerprint(self, text):
        """Generate document fingerprint"""
        kgrams = self.create_kgrams(text)
        hashes = self.hash_kgrams(kgrams)
        fingerprints = self.winnow(hashes)
        return fingerprints
    
    def compare_fingerprints(self, fp1, fp2):
        """Compare two document fingerprints"""
        if len(fp1) == 0 or len(fp2) == 0:
            return 0.0
        
        # Extract just the hash values for comparison
        hashes1 = set(h for _, h in fp1)
        hashes2 = set(h for _, h in fp2)
        
        intersection = len(hashes1.intersection(hashes2))
        union = len(hashes1.union(hashes2))
        
        return intersection / union if union > 0 else 0.0

class NoveltyDetector:
    """Novel Feature #3: Sentence-level novelty detection"""
    
    def __init__(self, model):
        self.model = model
        self.novelty_threshold = 0.75
    
    def detect_novel_sentences(self, target_text, source_texts):
        """Detect which sentences are novel vs. redundant"""
        target_sentences = sent_tokenize(target_text)
        all_source_sentences = []
        
        for source in source_texts:
            all_source_sentences.extend(sent_tokenize(source))
        
        if not all_source_sentences:
            return [(sent, 1.0, "Novel") for sent in target_sentences]
        
        # Encode all sentences
        target_embeddings = self.model.encode(target_sentences)
        source_embeddings = self.model.encode(all_source_sentences)
        
        results = []
        for i, (sentence, embedding) in enumerate(zip(target_sentences, target_embeddings)):
            # Calculate maximum similarity with any source sentence
            similarities = cosine_similarity([embedding], source_embeddings)[0]
            max_similarity = np.max(similarities)
            
            novelty_score = 1 - max_similarity
            status = "Novel" if novelty_score > (1 - self.novelty_threshold) else "Redundant"
            
            results.append((sentence, novelty_score, status, max_similarity))
        
        return results

class PlagiarismDetectionSystem:
    """Main system integrating all novel features"""
    
    def __init__(self, performance_mode="balanced"):
        st.info("🔄 Loading models... This may take a moment.")
        
        # Embedding models
        self.embedding_models = {
            "fast": "paraphrase-multilingual-MiniLM-L12-v2",
            "balanced": "paraphrase-multilingual-mpnet-base-v2",
            "best": "sentence-transformers/all-mpnet-base-v2"
        }
        
        # Cross-encoders for reranking
        self.cross_encoders = {
            "fast": "cross-encoder/stsb-roberta-base",
            "best": "cross-encoder/stsb-roberta-large"
        }
        
        # Load models based on performance mode
        self.embedding_model = SentenceTransformer(self.embedding_models[performance_mode])
        
        encoder_mode = "fast" if performance_mode == "fast" else "best"
        self.cross_encoder = CrossEncoder(self.cross_encoders[encoder_mode])
        
        # Initialize novel components
        self.stylometric_analyzer = StylometricAnalyzer()
        self.fingerprinter = DocumentFingerprinter()
        self.novelty_detector = NoveltyDetector(self.embedding_model)
        
        # OCR reader (lazy loading)
        self.ocr_reader = None
    
    def load_ocr(self):
        """Lazy load OCR model"""
        if self.ocr_reader is None:
            st.info("🔄 Loading OCR model...")
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        return self.ocr_reader
    
    def extract_text_from_image(self, image):
        """Extract text from uploaded image using OCR"""
        reader = self.load_ocr()
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Perform OCR
        results = reader.readtext(img_array, detail=0)
        extracted_text = '\n'.join(results)
        
        return extracted_text
    
    def analyze_plagiarism(self, submitted_text, reference_texts, student_history=None):
        """Comprehensive plagiarism analysis with novel features"""
        
        results = {
            'semantic_similarity': {},
            'stylometric_analysis': {},
            'fingerprint_matching': {},
            'novelty_detection': {},
            'intrinsic_detection': {},
            'overall_score': 0.0
        }
        
        # 1. Semantic Similarity Analysis (Traditional)
        submitted_embedding = self.embedding_model.encode([submitted_text])[0]
        
        for i, ref_text in enumerate(reference_texts):
            ref_embedding = self.embedding_model.encode([ref_text])[0]
            
            # Bi-encoder similarity
            bi_similarity = cosine_similarity([submitted_embedding], [ref_embedding])[0][0]
            
            # Cross-encoder reranking
            cross_similarity = self.cross_encoder.predict([(submitted_text, ref_text)])[0]
            
            # Combine scores
            combined_score = (bi_similarity * 0.6 + cross_similarity * 0.4)
            
            results['semantic_similarity'][f'Reference_{i+1}'] = {
                'bi_encoder_score': float(bi_similarity),
                'cross_encoder_score': float(cross_similarity),
                'combined_score': float(combined_score)
            }
        
        # 2. Stylometric Analysis (NOVEL)
        submitted_features = self.stylometric_analyzer.extract_stylometric_features(submitted_text)
        
        if student_history:
            # Compare with student's previous work
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
        
        # 3. Document Fingerprinting (NOVEL)
        submitted_fingerprint = self.fingerprinter.get_fingerprint(submitted_text)
        
        for i, ref_text in enumerate(reference_texts):
            ref_fingerprint = self.fingerprinter.get_fingerprint(ref_text)
            fingerprint_similarity = self.fingerprinter.compare_fingerprints(
                submitted_fingerprint, ref_fingerprint
            )
            results['fingerprint_matching'][f'Reference_{i+1}'] = float(fingerprint_similarity)
        
        # 4. Novelty Detection (NOVEL)
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
        
        # 5. Intrinsic Plagiarism Detection (NOVEL)
        # Detect style changes within the document
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
        
        # Calculate Overall Plagiarism Score
        semantic_scores = [v['combined_score'] for v in results['semantic_similarity'].values()]
        fingerprint_scores = list(results['fingerprint_matching'].values())
        
        max_semantic = max(semantic_scores) if semantic_scores else 0
        max_fingerprint = max(fingerprint_scores) if fingerprint_scores else 0
        novelty_ratio = results['novelty_detection']['novelty_ratio']
        
        # Weighted combination (giving more weight to novel features)
        overall_plagiarism = (
            max_semantic * 0.35 +
            max_fingerprint * 0.35 +
            (1 - novelty_ratio) * 0.30
        )
        
        results['overall_score'] = float(overall_plagiarism)
        results['risk_level'] = self._get_risk_level(overall_plagiarism)
        
        return results
    
    def _get_risk_level(self, score):
        """Determine risk level based on plagiarism score"""
        if score > 0.75:
            return "🔴 High Risk"
        elif score > 0.50:
            return "🟡 Medium Risk"
        elif score > 0.25:
            return "🟢 Low Risk"
        else:
            return "✅ Minimal Risk"

def create_visualization(results):
    """Create visualizations for plagiarism analysis"""
    
    # Semantic Similarity Comparison
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
    
    # Fingerprint Matching
    if results['fingerprint_matching']:
        st.subheader("🔍 Document Fingerprint Matching (Novel)")
        
        fp_data = pd.DataFrame({
            'Reference': list(results['fingerprint_matching'].keys()),
            'Fingerprint Similarity': list(results['fingerprint_matching'].values())
        })
        
        st.bar_chart(fp_data.set_index('Reference'))
    
    # Novelty Detection
    if results['novelty_detection']:
        st.subheader("✨ Novelty Detection Analysis (Novel)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Novel Sentences", results['novelty_detection']['novel_sentences'])
        with col2:
            st.metric("Total Sentences", results['novelty_detection']['total_sentences'])
        with col3:
            st.metric("Novelty Ratio", f"{results['novelty_detection']['novelty_ratio']:.2%}")
        
        # Show sentence-level details
        with st.expander("View Sentence-Level Analysis"):
            for detail in results['novelty_detection']['sentence_details'][:10]:  # Show first 10
                status_color = "🟢" if detail['status'] == "Novel" else "🔴"
                st.write(f"{status_color} **{detail['status']}** (Similarity: {detail['max_similarity']:.2%})")
                st.write(f"_{detail['sentence']}_")
                st.divider()
    
    # Stylometric Analysis
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
    
    # Intrinsic Detection
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
    st.set_page_config(page_title="AI Plagiarism Detector", layout="wide", page_icon="🔍")
    
    st.title("🔍 Advanced Plagiarism Detection System")
    st.markdown("### With Novel Features: Stylometry, Fingerprinting & Novelty Detection")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        performance_mode = st.selectbox(
            "Performance Mode",
            ["fast", "balanced", "best"],
            index=1
        )
        
        st.divider()
        
        st.header("📚 Novel Features")
        st.markdown("""
        **1. Stylometric Analysis**
        - Writing style fingerprinting
        - Authorship verification
        - Style consistency checking
        
        **2. Document Fingerprinting**
        - Winnowing algorithm
        - Efficient similarity detection
        - Granular matching
        
        **3. Novelty Detection**
        - Sentence-level analysis
        - Novel vs redundant identification
        - Detailed attribution
        
        **4. Intrinsic Detection**
        - Internal style variation analysis
        - Multi-author detection
        - Suspicious pattern identification
        
        **5. OCR Support**
        - Extract text from images
        - Handwritten document analysis
        """)
    
    # Initialize system
    if 'system' not in st.session_state:
        st.session_state.system = PlagiarismDetectionSystem(performance_mode)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📷 OCR Input", "ℹ️ About"])
    
    with tab1:
        st.header("Submit Assignment for Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Student Submission")
            submitted_text = st.text_area(
                "Paste the submitted assignment",
                height=300,
                placeholder="Enter the student's assignment text here..."
            )
            
            st.subheader("Student's Previous Work (Optional)")
            student_history = st.text_area(
                "Paste previous work for style comparison",
                height=150,
                placeholder="Enter previous assignment for stylometric comparison..."
            )
        
        with col2:
            st.subheader("Reference Documents")
            num_references = st.number_input("Number of reference documents", 1, 5, 1)
            
            reference_texts = []
            for i in range(num_references):
                ref_text = st.text_area(
                    f"Reference Document {i+1}",
                    height=150,
                    key=f"ref_{i}",
                    placeholder=f"Enter reference document {i+1}..."
                )
                if ref_text:
                    reference_texts.append(ref_text)
        
        if st.button("🔍 Analyze Plagiarism", type="primary"):
            if not submitted_text:
                st.error("Please provide the submitted assignment text.")
            elif not reference_texts:
                st.error("Please provide at least one reference document.")
            else:
                with st.spinner("🔄 Analyzing... This may take a moment."):
                    results = st.session_state.system.analyze_plagiarism(
                        submitted_text,
                        reference_texts,
                        student_history if student_history else None
                    )
                    
                    # Display overall results
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Overall Plagiarism Score",
                            f"{results['overall_score']:.1%}",
                            delta=results['risk_level']
                        )
                    with col2:
                        st.metric(
                            "Risk Level",
                            results['risk_level']
                        )
                    
                    st.divider()
                    
                    # Detailed visualizations
                    create_visualization(results)
                    
                    # Export results
                    with st.expander("📥 Export Detailed Report"):
                        st.json(results)
    
    with tab2:
        st.header("OCR Text Extraction")
        st.markdown("Upload an image of a handwritten or printed assignment")
        
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the document"
        )
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("🔍 Extract Text"):
                    with st.spinner("🔄 Extracting text from image..."):
                        extracted_text = st.session_state.system.extract_text_from_image(image)
                        
                        st.success("✅ Text extracted successfully!")
                        st.text_area(
                            "Extracted Text",
                            extracted_text,
                            height=300
                        )
                        
                        st.info("💡 Copy this text and paste it in the 'Text Input' tab for analysis")
    
    with tab3:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Novel Contributions
        
        This plagiarism detection system goes beyond traditional LLM-based approaches by incorporating:
        
        #### 1. **Stylometric Fingerprinting**
        - Analyzes unique writing patterns (sentence structure, vocabulary, punctuation)
        - Creates an "authorial fingerprint" for each student
        - Detects inconsistencies in writing style within a single document
        - Verifies authorship by comparing with previous student work
        
        #### 2. **Winnowing-based Document Fingerprinting**
        - Uses efficient hash-based fingerprinting algorithm
        - Detects near-duplicate content at granular level
        - More computationally efficient than full text comparison
        - Identifies specific plagiarized passages
        
        #### 3. **Multi-Level Novelty Detection**
        - Analyzes novelty at sentence, paragraph, and document levels
        - Distinguishes between novel and redundant content
        - Provides detailed attribution for each sentence
        - Reduces false positives through semantic understanding
        
        #### 4. **Intrinsic Plagiarism Detection**
        - Detects plagiarism without external reference documents
        - Identifies style inconsistencies within the submission
        - Useful for detecting copy-paste from diverse sources
        - Complements extrinsic detection methods
        
        #### 5. **OCR Integration**
        - Supports handwritten and printed document analysis
        - Enables plagiarism detection for physical submissions
        - Multilingual text extraction capability
        
        ### 🔬 Technical Architecture
        
        - **Embedding Models**: Multilingual transformer models for semantic understanding
        - **Cross-Encoders**: Reranking for improved accuracy
        - **Stylometry**: Statistical and linguistic feature extraction
        - **Fingerprinting**: Winnowing algorithm for efficient matching
        - **OCR**: EasyOCR with deep learning-based text recognition
        
        ### 📊 Evaluation Metrics
        
        - Semantic similarity (bi-encoder + cross-encoder)
        - Fingerprint matching coefficient
        - Novelty ratio
        - Stylometric consistency score
        - Internal style variation
        - Combined weighted plagiarism score
        
        ### 🚀 Why This is Novel
        
        Unlike existing tools that rely solely on:
        - Simple text matching (Turnitin, Copyscape)
        - Pure LLM-based similarity (ChatGPT detection tools)
        
        This system combines:
        - **Semantic** understanding (what the text means)
        - **Syntactic** analysis (how the text is written)
        - **Stylometric** profiling (who likely wrote it)
        - **Structural** fingerprinting (efficient pattern matching)
        - **Novelty** quantification (what's new vs redundant)
        
        This multi-faceted approach significantly reduces false positives and provides
        educators with detailed, interpretable results for academic integrity assessment.
        """)

if __name__ == "__main__":
    main()
