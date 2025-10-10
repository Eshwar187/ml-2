import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
from docx import Document
import tempfile
import os
import ssl
import hashlib
from PIL import Image
import easyocr
from langdetect import detect, LangDetectException

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

class PlagiarismDetectorV4:
    def __init__(self):
        with st.spinner("🔄 Loading models..."):
            self.semantic_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', device='cpu')
            self.paraphrase_model = SentenceTransformer('distiluse-base-multilingual-cased-v2', device='cpu')
        self.ocr_reader = None
    
    def load_ocr(self):
        if self.ocr_reader is None:
            try:
                with st.spinner("🔄 Loading OCR (first time only)..."):
                    
                    langs = ['en', 'hi', 'ar', 'ch_sim', 'ta', 'te']  # ch_sim NOT zh_sim!
                    self.ocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)
                    st.success(f"✅ OCR loaded for: {', '.join(langs)}")
            except Exception as e:
                st.error(f"❌ OCR failed: {str(e)}")
                st.info("Continuing without OCR. You can still use PDF/DOCX/text.")
                return None
        return self.ocr_reader

    
    def detect_language(self, text):
        """Detect language of text"""
        try:
            return detect(text)
        except:
            return "unknown"
    
    def get_text_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_text_from_pdf(self, pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"PDF: {str(e)}")
            return ""
    
    def get_text_from_docx(self, docx_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                tmp.write(docx_file.read())
                tmp.flush()
                tmp_path = tmp.name
            
            docx_file.seek(0)
            doc = Document(tmp_path)
            text = '\n'.join([p.text for p in doc.paragraphs if p.text])
            
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            return text.strip()
        except Exception as e:
            st.error(f"DOCX: {str(e)}")
            return ""
    
    def get_text_from_image(self, image_file):
        try:
            reader = self.load_ocr()
            image = Image.open(image_file)
            results = reader.readtext(np.array(image), detail=0)
            return '\n'.join(results) if results else ""
        except Exception as e:
            st.error(f"OCR: {str(e)}")
            return ""
    
    def extract_from_file(self, uploaded_file):
        if not uploaded_file:
            return ""
        
        try:
            file_type = uploaded_file.type
            
            if file_type == "application/pdf":
                return self.get_text_from_pdf(uploaded_file)
            elif "wordprocessing" in file_type:
                return self.get_text_from_docx(uploaded_file)
            elif file_type in ["image/png", "image/jpeg", "image/jpg"]:
                return self.get_text_from_image(uploaded_file)
            else:
                st.warning(f"Unsupported: {file_type}")
                return ""
        except Exception as e:
            st.error(f"Failed: {str(e)}")
            return ""
    
    def detect_plagiarism(self, submission, references, plag_threshold=0.70, exact_threshold=0.95, cross_lingual_penalty=0.15):
        try:
            sub_sents = sent_tokenize(submission)
        except:
            sub_sents = [s.strip() for s in submission.split('.') if s.strip()]
        
        if not sub_sents or not references:
            return self._empty()
        
        # Detect submission language
        sub_lang = self.detect_language(submission)
        st.info(f"📝 Detected submission language: **{sub_lang.upper()}**")
        
        # Check for identical documents
        sub_hash = self.get_text_hash(submission.strip())
        for idx, ref in enumerate(references):
            ref_hash = self.get_text_hash(ref.strip())
            if sub_hash == ref_hash:
                st.warning(f"⚠️ WARNING: Submission is identical to Reference {idx+1}!")
        
        ref_sents = []
        sources = []
        ref_langs = []
        
        for idx, ref in enumerate(references):
            ref_lang = self.detect_language(ref)
            st.info(f"📚 Reference {idx+1} language: **{ref_lang.upper()}**")
            
            try:
                sents = sent_tokenize(ref)
            except:
                sents = [s.strip() for s in ref.split('.') if s.strip()]
            for s in sents:
                if len(s) > 15:
                    ref_sents.append(s)
                    sources.append(f"Ref {idx+1}")
                    ref_langs.append(ref_lang)
        
        if not ref_sents:
            return self._empty()
        
        # Warning for cross-lingual comparison
        if sub_lang != "unknown":
            different_langs = [rl for rl in ref_langs if rl != sub_lang and rl != "unknown"]
            if different_langs:
                st.warning(f"⚠️ **Cross-lingual detection active**: Submission ({sub_lang}) vs Reference ({', '.join(set(different_langs))}). Applying {cross_lingual_penalty*100:.0f}% penalty to reduce false positives.")
        
        st.info(f"🔍 Analyzing {len(sub_sents)} sentences...")
        
        with st.spinner("Encoding..."):
            sub_emb1 = self.semantic_model.encode(sub_sents, show_progress_bar=False)
            ref_emb1 = self.semantic_model.encode(ref_sents, show_progress_bar=False)
            sub_emb2 = self.paraphrase_model.encode(sub_sents, show_progress_bar=False)
            ref_emb2 = self.paraphrase_model.encode(ref_sents, show_progress_bar=False)
        
        details = []
        plag_count = 0
        exact_count = 0
        
        prog = st.progress(0)
        
        for i, sent in enumerate(sub_sents):
            prog.progress((i+1)/len(sub_sents))
            
            sims1 = cosine_similarity([sub_emb1[i]], ref_emb1)[0]
            sims2 = cosine_similarity([sub_emb2[i]], ref_emb2)[0]
            
            idx1 = int(np.argmax(sims1))
            idx2 = int(np.argmax(sims2))
            
            score1 = float(sims1[idx1])
            score2 = float(sims2[idx2])
            
            # Use average of both models
            avg_score = (score1 + score2) / 2
            
            matched_idx = idx1 if score1 > score2 else idx2
            matched_lang = ref_langs[matched_idx]
            
            # Apply cross-lingual penalty if languages differ
            final_score = avg_score
            if sub_lang != matched_lang and sub_lang != "unknown" and matched_lang != "unknown":
                final_score = avg_score * (1 - cross_lingual_penalty)
            
            is_exact = final_score >= exact_threshold
            is_plag = final_score >= plag_threshold
            
            if is_plag:
                plag_count += 1
                if is_exact:
                    exact_count += 1
            
            details.append({
                'num': i+1,
                'text': sent,
                'plag': is_plag,
                'exact': is_exact,
                'score': final_score,
                'raw_score': avg_score,
                'match': ref_sents[matched_idx],
                'source': sources[matched_idx],
                'cross_lingual': sub_lang != matched_lang
            })
        
        prog.empty()
        
        total = len(sub_sents)
        plag_pct = (plag_count / total * 100) if total > 0 else 0
        
        return {
            'plagiarism': float(plag_pct),
            'originality': float(100 - plag_pct),
            'total': int(total),
            'plagiarized': int(plag_count),
            'exact': int(exact_count),
            'original': int(total - plag_count),
            'details': details
        }
    
    def _empty(self):
        return {
            'plagiarism': 0.0,
            'originality': 100.0,
            'total': 0,
            'plagiarized': 0,
            'exact': 0,
            'original': 0,
            'details': []
        }

@st.cache_resource(show_spinner=False)
def get_detector():
    return PlagiarismDetectorV4()

def main():
    st.title("🌍 Multilingual Plagiarism Detector")
    st.markdown("### AI-Powered • OCR • 100+ Languages • Cross-Lingual Detection")
    
    with st.expander("📖 HOW TO USE & UNDERSTAND RESULTS"):
        st.markdown("""
        **✅ CORRECT USAGE:**
        - **Submission:** Student's assignment
        - **Reference:** Original sources to check against
        
        **🔍 CROSS-LINGUAL DETECTION:**
        - This system detects **semantic similarity** across languages
        - If Telugu text and Tamil text discuss the SAME TOPIC, they will show high similarity
        - **This is correct behavior!** It prevents translated plagiarism
        
        **Example:**
        - Telugu: "కృత్రిమ మేధస్సు అనేది..." (About AI)
        - Tamil: "செயற்கை நுண்ண றிவு என்பது..." (About AI)
        - **Result:** High similarity (same meaning, different language)
        
        **💡 TIP:** Use the threshold sliders below to adjust sensitivity
        """)
    
    detector = get_detector()
    
    # Adjustable thresholds
    st.sidebar.header("⚙️ Settings")
    plag_thresh = st.sidebar.slider("Plagiarism Threshold", 0.5, 0.95, 0.70, 0.05)
    exact_thresh = st.sidebar.slider("Exact Copy Threshold", 0.85, 1.0, 0.95, 0.05)
    cross_penalty = st.sidebar.slider("Cross-Lingual Penalty", 0.0, 0.3, 0.15, 0.05)
    
    st.sidebar.markdown(f"""
    **Current Settings:**
    - Plagiarism: ≥ {plag_thresh*100:.0f}%
    - Exact Copy: ≥ {exact_thresh*100:.0f}%
    - Cross-Lingual Penalty: {cross_penalty*100:.0f}%
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Submission")
        st.caption("Document to CHECK")
        sub_file = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key="sub")
        
        sub_text = ""
        if sub_file:
            with st.spinner("Extracting..."):
                sub_text = detector.extract_from_file(sub_file)
            if sub_text:
                st.success(f"✅ {len(sub_text)} chars | {sub_file.name}")
        
        sub_input = st.text_area("Or paste text", value=sub_text, height=300)
    
    with col2:
        st.subheader("📚 References")
        st.caption("Sources to CHECK AGAINST")
        num_refs = st.number_input("Number", 1, 5, 1)
        
        ref_texts = []
        
        for i in range(num_refs):
            st.markdown(f"**Reference {i+1}**")
            ref_file = st.file_uploader("Upload", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'], key=f"ref{i}")
            
            ref_text = ""
            if ref_file:
                with st.spinner("Extracting..."):
                    ref_text = detector.extract_from_file(ref_file)
                if ref_text:
                    st.success(f"✅ {len(ref_text)} chars | {ref_file.name}")
                    ref_texts.append(ref_text)
            
            ref_input = st.text_area("Or paste", value=ref_text, height=100, key=f"rt{i}")
            
            if ref_input and ref_input.strip() and not ref_file:
                ref_texts.append(ref_input.strip())
    
    if st.button("🔍 DETECT PLAGIARISM", type="primary"):
        if not sub_input or not sub_input.strip():
            st.error("❌ Provide submission")
        elif not ref_texts:
            st.error("❌ Provide references")
        else:
            results = detector.detect_plagiarism(sub_input, ref_texts, plag_thresh, exact_thresh, cross_penalty)
            
            st.success("✅ ANALYSIS COMPLETE!")
            
            plag = results['plagiarism']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Plagiarism", f"{'🔴' if plag>30 else '🟠' if plag>15 else '🟢'} {plag:.1f}%")
            with col2:
                st.metric("Originality", f"{results['originality']:.1f}%")
            with col3:
                st.metric("Plagiarized", f"{results['plagiarized']}/{results['total']}")
            with col4:
                st.metric("Exact", f"{results['exact']}")
            
            st.divider()
            
            exact = [d for d in results['details'] if d['exact']]
            plag_items = [d for d in results['details'] if d['plag'] and not d['exact']]
            
            if exact:
                st.error(f"🔴 {len(exact)} EXACT COPIES")
                for item in exact[:3]:
                    st.markdown(f"**{item['num']}.** {item['text']}")
                    cross_info = " (Cross-lingual)" if item['cross_lingual'] else ""
                    st.caption(f"Score: {item['score']:.2%} (Raw: {item['raw_score']:.2%}){cross_info} | {item['source']}")
                    st.info(f"Match: {item['match'][:100]}...")
                    st.divider()
            
            if plag_items:
                with st.expander(f"🟠 Plagiarized ({len(plag_items)})"):
                    for item in plag_items[:10]:
                        cross_info = " 🌐" if item['cross_lingual'] else ""
                        st.markdown(f"**{item['num']}.{cross_info}** {item['text']}")
                        st.caption(f"Score: {item['score']:.2%} | {item['source']}")

if __name__ == "__main__":
    main()
