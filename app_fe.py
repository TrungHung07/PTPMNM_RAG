import streamlit as st
import requests
import uuid
import time
import os
import re
from pathlib import Path
from typing import List, Optional

# --- Configuration ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Gemini Aesthetics ---
st.markdown("""
<style>
    /* ── Google Sans font ── */
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Roboto:wght@300;400;500&display=swap');

    *, *::before, *::after {
        box-sizing: border-box;
    }

    /* Áp dụng font chữ an toàn */
    html, body, .stApp, .stMarkdown, p, h1, h2, h3, span, div, button, input {
        font-family: 'Google Sans', 'Roboto', sans-serif;
    }

    /* ── App background ── */
    .stApp { background-color: #FFFFFF; }
    .block-container { padding-top: 2rem !important; padding-bottom: 6rem !important; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E8EAED !important;
    }
    
    .sd-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        padding-bottom: 1.25rem;
        color: #202124;
        font-size: 1.25rem;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    .sd-logo-gem {
        background: linear-gradient(135deg, #007BFF 0%, #0056CC 100%);
        color: #fff;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        flex-shrink: 0;
        box-shadow: 0 2px 8px rgba(0,123,255,0.35);
    }

    /* Sidebar section labels */
    .sd-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #80868B;
        text-transform: uppercase;
        letter-spacing: 0.9px;
        margin: 1.1rem 0 0.45rem 0;
        display: block;
    }
    .sd-divider {
        border: none;
        border-top: 1px solid #E8EAED;
        margin: 0.9rem 0;
    }

    /* File cards in sidebar */
    .file-card {
        display: flex;
        align-items: flex-start;
        gap: 9px;
        background: #FFFFFF;
        border: 1px solid #E8EAED;
        border-left: 3px solid #007BFF;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.45rem;
    }
    .file-name { font-weight: 500; font-size: 0.84rem; color: #202124; word-break: break-all; }
    .file-badge {
        display: inline-block;
        background: #E6F4EA;
        color: #34A853;
        font-size: 0.7rem;
        font-weight: 600;
        border-radius: 20px;
        padding: 1px 7px;
        margin-top: 3px;
    }

    /* Chat Messages - Hide Avatars Aggressively */
    [data-testid="stChatMessageAvatar"], 
    .stChatMessageAvatar,
    div[data-testid*="chatAvatarIcon"],
    [data-testid*="Avatar"],
    div[class*="Avatar"] { 
        display: none !important; 
    }
    
    [data-testid="stChatMessage"] {
        background: transparent !important;
        max-width: 850px !important;
        margin: 0 auto !important;
    }

    /* ── Custom HTML Expander for Citations ── */
    details.custom-expander {
        border: 1px solid #E0ECFF;
        border-radius: 12px;
        background: #FAFCFF;
        margin-top: 0.6rem;
        overflow: hidden;
        padding: 12px 16px;
    }
    details.custom-expander summary {
        color: #007BFF;
        font-weight: 500;
        font-size: 0.875rem;
        padding: 0.65rem 1rem;
        cursor: pointer;
        list-style: none;
    }
    .cit-header {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EFF6FF;
        border: 1px solid #BDD7FF;
        border-radius: 20px;
        padding: 3px 10px;
        margin-bottom: 0.55rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: #0056CC;
    }
    .cit-body {
        background: #F8F9FA;
        border-left: 3px solid #007BFF;
        border-radius: 0 10px 10px 0;
        padding: 0.75rem 1rem;
        font-style: italic;
        font-size: 0.875rem;
        line-height: 1.65;
        color: #3C4043;
    }

    /* Comparison Cards */
    .comp-card {
        background-color: white;
        border: 1px solid #dadce0;
        border-radius: 12px;
        margin-bottom: 10px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        height: 100%;
    }
    .comp-header {
        padding: 10px 16px;
        font-weight: 600;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid #dadce0;
    }
    .vector-header { background-color: #e8f0fe; color: #1967d2; }
    .hybrid-header { background-color: #e6f4ea; color: #137333; }
    .comp-content { padding: 16px; font-size: 0.95rem; line-height: 1.6; flex-grow: 1; }

    /* Metrics & Comparison Details */
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-top: 12px;
        padding-top: 8px;
        border-top: 1px dashed #e0e0e0;
        font-size: 0.75rem;
        color: #5F6368;
    }
    .metric-item {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    .metric-value {
        font-weight: 600;
        color: #202124;
    }
    .source-summary {
        margin-top: 12px;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 8px;
        font-size: 0.8rem;
        border: 1px solid #e8eaed;
    }
    .source-title {
        font-weight: 600;
        color: #5F6368;
        font-size: 0.7rem;
        text-transform: uppercase;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    header { visibility: hidden; }
    header button { visibility: visible !important; }

    /* Borderless Sidebar Settings Button */
    .settings-wrapper {
        margin-top: auto;
        padding-top: 2rem;
    }
    .settings-wrapper button {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: #5F6368 !important;
        text-align: left !important;
        padding-left: 0.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .settings-wrapper button:hover {
        background-color: #E8F0FE !important;
        color: #007BFF !important;
    }

    /* Make st.file_uploader button full width */
    [data-testid="stFileUploader"] section button {
        width: 100% !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
    }
    /* Highlight styling */
    mark {
        background-color: #FFF2CC !important;
        color: #202124 !important;
        padding: 0 2px !important;
        border-radius: 2px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "file_ids_map" not in st.session_state:
    # {filename: {"id": file_id, "type": type}}
    st.session_state.file_ids_map = {}
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_files" not in st.session_state:
    st.session_state.selected_files = []
if "available_sessions" not in st.session_state:
    st.session_state.available_sessions = []
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "hybrid"
if "bm25_weight" not in st.session_state:
    st.session_state.bm25_weight = 0.5
if "eval_mode" not in st.session_state:
    st.session_state.eval_mode = False
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 200
if "rerank_enabled" not in st.session_state:
    st.session_state.rerank_enabled = True
if "rerank_threshold" not in st.session_state:
    st.session_state.rerank_threshold = 0.1

@st.dialog("Cấu hình hệ thống")
def settings_dialog():
    st.markdown("##### Cấu hình Chunking")
    st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size, 100)
    st.session_state.chunk_overlap = st.slider("Overlap", 0, 500, st.session_state.chunk_overlap, 50)
    
    st.divider()
    
    st.markdown("##### Chế độ tìm kiếm")
    st.session_state.eval_mode = st.toggle("Chế độ so sánh (Hybrid vs Vector)", value=st.session_state.eval_mode)
    
    if not st.session_state.eval_mode:
        st.session_state.search_mode = st.radio(
            "Chiến lược retrieval",
            options=["hybrid", "vector"],
            index=0 if st.session_state.search_mode == "hybrid" else 1,
            horizontal=True
        )
        if st.session_state.search_mode == "hybrid":
            st.session_state.bm25_weight = st.slider(
                "Trọng số BM25", 0.0, 1.0, st.session_state.bm25_weight, 0.1,
                help="Tăng để ưu tiên từ khóa chính xác, giảm để ưu tiên ngữ nghĩa."
            )
            
    st.divider()
    st.markdown("##### Tinh chỉnh độ chính xác (Reranking)")
    st.session_state.rerank_enabled = st.toggle("Bật Reranking (Cross-Encoder)", value=st.session_state.rerank_enabled)
    if st.session_state.rerank_enabled:
        st.session_state.rerank_threshold = st.slider(
            "Độ khắt khe của nguồn trích dẫn", 0.0, 1.0, st.session_state.rerank_threshold, 0.05,
            help="Model PhoRanker dùng thang điểm 0-1. Tăng lên 0.2-0.5 để lọc bỏ hoàn toàn các đoạn không liên quan."
        )
            
    
    if st.button("Đóng", type="primary", use_container_width=True):
        st.rerun()

# --- API Helpers ---
class API:
    @staticmethod
    def get_sessions():
        try:
            resp = requests.get(f"{API_BASE_URL}/history")
            if resp.status_code == 200:
                return resp.json()
        except: pass
        return []

    @staticmethod
    def get_session_history(sid):
        try:
            resp = requests.get(f"{API_BASE_URL}/history/{sid}")
            if resp.status_code == 200:
                return resp.json()
        except: pass
        return None

    @staticmethod
    def upload(files, c_size, c_overlap):
        files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        resp = requests.post(
            f"{API_BASE_URL}/upload", 
            params={"chunk_size": c_size, "chunk_overlap": c_overlap},
            files=files_payload
        )
        return resp.json() if resp.status_code == 200 else None

    @staticmethod
    def ask(qid, sid, fids, mode, weight, r_enabled, r_threshold):
        payload = {
            "session_id": sid,
            "file_ids": fids,
            "question": qid,
            "search_mode": mode,
            "bm25_weight": weight,
            "rerank_enabled": r_enabled,
            "rerank_threshold": r_threshold
        }
        resp = requests.post(f"{API_BASE_URL}/ask", json=payload)
        return resp.json() if resp.status_code == 200 else None

    @staticmethod
    def compare(qid, sid, fids, weight):
        payload = {
            "session_id": sid,
            "file_ids": fids,
            "question": qid,
            "bm25_weight": weight
        }
        resp = requests.post(f"{API_BASE_URL}/compare", json=payload)
        return resp.json() if resp.status_code == 200 else None

    @staticmethod
    def delete_session(sid):
        try:
            resp = requests.delete(f"{API_BASE_URL}/history/{sid}")
            return resp.status_code == 200
        except: return False

    @staticmethod
    def clear_all_history():
        try:
            resp = requests.delete(f"{API_BASE_URL}/history")
            return resp.status_code == 200
        except: return False

# --- Helper Functions ---
def render_citations(sources: list, query: str = None):
    """Render citation with optional keyword highlighting."""
    if not sources:
        return
        
    # Extract keywords for highlighting
    keywords = []
    if query:
        # Tách từ, bỏ các ký tự đặc biệt, lọc từ ngắn < 2 ký tự
        raw_words = re.findall(r'\w+', query.lower())
        keywords = [w for w in raw_words if len(w) > 1]
        
    citations_html = ""
    for i, src in enumerate(sources, start=1):
        page = src.get('metadata', {}).get('page', src.get('metadata', {}).get('paragraph', '?'))
        content = src.get('content', '')
        
        # Thực hiện highlight
        if keywords:
            # Tạo regex pattern cho tất cả keywords (case-insensitive)
            # Sắp xếp keyword dài lên trước để tránh highlight đè (vd: "chạy bộ" trước "chạy")
            keywords_sorted = sorted(keywords, key=len, reverse=True)
            pattern = re.compile(f'({"|".join(map(re.escape, keywords_sorted))})', re.IGNORECASE)
            content = pattern.sub(r'<mark>\1</mark>', content)

        score = src.get('score')
        score_html = f' &nbsp;·&nbsp; <span style="color:#007BFF; font-weight:bold;">Điểm: {score:.2f}</span>' if score is not None else ""
        citations_html += f'<div class="cit-header">📄 Đoạn {i} &nbsp;·&nbsp; Trang {page}{score_html}</div><div class="cit-body">{content}</div>'
        if i < len(sources):
            citations_html += "<hr style='border:none;border-top:1px solid #E8EAED;margin:0.3rem 0 0.7rem 0'>"
            
    html_block = f"""<details class="custom-expander">
<summary>Nguồn từ tài liệu</summary>
<div class="custom-expander-content">
<p style='color:#5F6368;font-size:0.82rem;margin-bottom:0.8rem;margin-top:0.5rem;'>Các đoạn văn bản gốc mà AI sử dụng để trả lời câu hỏi của bạn:</p>
{citations_html}
</div>
</details>"""
    
    st.markdown(html_block, unsafe_allow_html=True)

# --- Lifecycle Logic ---
def load_session_list():
    st.session_state.available_sessions = API.get_sessions()

def switch_session(sid):
    with st.spinner("Đang tải dữ liệu phiên cũ..."):
        data = API.get_session_history(sid)
        if data:
            st.session_state.session_id = sid
            # Restore files
            new_map = {}
            for f in data["files"]:
                new_map[f["file_name"]] = {"id": f["file_id"], "type": f["file_type"]}
            st.session_state.file_ids_map = new_map
            st.session_state.selected_files = list(new_map.keys())
            
            # Build file_id -> file_name lookup
            fid_to_name = {v["id"]: fname for fname, v in new_map.items()}
            # Restore messages
            st.session_state.messages = []
            for m in data["history"]:
                file_ids = m.get("file_ids", [])
                file_names = [fid_to_name[fid] for fid in file_ids if fid in fid_to_name]
                st.session_state.messages.append({
                    "role": "user",
                    "content": m["question"],
                    "file_names": file_names,
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": m["answer"],
                    "citations": m.get("citations") or [],
                    "search_mode": m.get("search_mode") or "unknown",
                })
            st.rerun()

# Initial load
if not st.session_state.available_sessions:
    load_session_list()

# --- SIDEBAR: Document Control ---
with st.sidebar:
    st.markdown(
        """
        <div class="sd-logo">
            <div class="sd-logo-gem">✦</div>
            SmartDoc AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. New Chat & Sessions
    if st.button("✦ &nbsp; Cuộc trò chuyện mới", use_container_width=True):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.file_ids_map = {}
        st.session_state.selected_files = []
        st.rerun()

    st.markdown('<span class="sd-label">Phiên hỏi đáp</span>', unsafe_allow_html=True)
    session_options = ["Chọn phiên làm việc"] + [
        f"{s['session_id'][:8]}... ({s['file_count']} file)" 
        for s in st.session_state.available_sessions
    ]
    
    current_idx = 0
    if st.session_state.session_id:
        for i, s in enumerate(st.session_state.available_sessions):
            if s['session_id'] == st.session_state.session_id:
                current_idx = i + 1
                break
                
    chosen_sess = st.selectbox("Chọn phiên", options=session_options, index=current_idx, label_visibility="collapsed")
    
    if st.session_state.session_id:
        if st.button("Xóa phiên hiện tại", help="Xóa phiên này", use_container_width=True):
            if API.delete_session(st.session_state.session_id):
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.file_ids_map = {}
                st.session_state.selected_files = []
                load_session_list()
                st.rerun()
            else:
                st.error("Không thể xóa phiên.")
    
    if chosen_sess != "Chọn phiên làm việc":
        sid = st.session_state.available_sessions[session_options.index(chosen_sess)-1]['session_id']
        if sid != st.session_state.session_id:
            switch_session(sid)
    
    if st.session_state.available_sessions:
        if st.button("Clear All History", use_container_width=True, type="secondary"):
            if API.clear_all_history():
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.file_ids_map = {}
                st.session_state.selected_files = []
                load_session_list()
                st.rerun()
            else:
                st.error("Không thể xóa lịch sử.")

    st.markdown('<hr class="sd-divider">', unsafe_allow_html=True)
    
    u_files = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"], accept_multiple_files=True, label_visibility="collapsed")
    if u_files and st.button("Phân tích", type="primary", use_container_width=True):
        with st.status("Đang xử lý tài liệu...") as status:
            res = API.upload(u_files, st.session_state.chunk_size, st.session_state.chunk_overlap)
            if res:
                st.session_state.session_id = res["session_id"]
                for f in res["files"]:
                    st.session_state.file_ids_map[f["filename"]] = {"id": f["file_id"], "type": "pdf"}
                st.session_state.selected_files = list(st.session_state.file_ids_map.keys())
                load_session_list()
                status.update(label="Tài liệu đã sẵn sàng!", state="complete", expanded=False)
                st.rerun()

    # 3. File Explorer
    if st.session_state.file_ids_map:
        st.markdown('<hr class="sd-divider">', unsafe_allow_html=True)
        st.markdown('<span class="sd-label">Tài liệu đã chọn</span>', unsafe_allow_html=True)
        for fname in st.session_state.file_ids_map.keys():
            is_sel = fname in st.session_state.selected_files
            
            # Align checkbox and filename horizontally
            col_check, col_name = st.columns([1, 9])
            with col_check:
                checked = st.checkbox(fname, value=is_sel, key=f"src_{fname}", label_visibility="collapsed")
            with col_name:
                st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 8px; background: white; border: 1px solid #E8EAED; border-left: 3px solid #007BFF; border-radius: 8px; padding: 4px 10px; margin-top: 2px;">
                        <span style="font-size: 1rem;">📄</span>
                        <div style="font-weight: 500; font-size: 0.85rem; color: #202124; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{fname}">{fname}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            if checked and fname not in st.session_state.selected_files:
                st.session_state.selected_files.append(fname)
            elif not checked and fname in st.session_state.selected_files:
                st.session_state.selected_files.remove(fname)

    st.markdown('<div class="settings-wrapper">', unsafe_allow_html=True)
    if st.button("Cấu hình hệ thống", key="sidebar_settings", use_container_width=True):
        settings_dialog()
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN: Chat Interface ---

# ── Welcome header (shown when no messages) ──
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 1rem 1.5rem 1rem;">
            <p style="font-size: 2.4rem; font-weight: 700; letter-spacing: -0.8px; background: linear-gradient(135deg, #007BFF 0%, #0056CC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 0.4rem 0;">SmartDoc AI</p>
            <p style="color: #5F6368; font-size: 1rem; font-weight: 400; margin: 0;">Đặt câu hỏi về tài liệu — tôi trả lời dựa trên nội dung thực tế, có nguồn trích dẫn.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    if st.session_state.selected_files:
        st.caption(f"Ngữ cảnh: {', '.join(st.session_state.selected_files)}")
    elif st.session_state.messages:
        st.caption("Vui lòng chọn tài liệu ở thanh bên để làm ngữ cảnh.")

with col_t2:
    pass

# Luồng tin nhắn
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            file_tags_html = ""
            msg_file_names = msg.get("file_names", [])
            if msg_file_names:
                tags = "".join(
                    f'<span style="display:inline-flex;align-items:center;gap:4px;background:#EFF6FF;border:1px solid #BDD7FF;border-radius:20px;padding:2px 9px;font-size:0.72rem;font-weight:600;color:#0056CC;">📄 {fn}</span>'
                    for fn in msg_file_names
                )
                file_tags_html = f'<div style="display:flex;justify-content:flex-end;flex-wrap:wrap;gap:4px;margin-top:6px;">{tags}</div>'
            st.markdown(f'''
                <div style="display: flex; flex-direction: column; align-items: flex-end; width: 100%;">
                    <div style="background-color: #F0F2F6; padding: 12px 18px; border-radius: 20px 20px 4px 20px; color: #202124; text-align: left; display: inline-block; max-width: 85%; font-size: 0.95rem; line-height: 1.5; word-wrap: break-word;">
                        {msg["content"]}
                    </div>
                    {file_tags_html}
                </div>
            ''', unsafe_allow_html=True)
        elif "compare_data" in msg:
            res_v = msg["compare_data"]["vector"]
            res_h = msg["compare_data"]["hybrid"]
            c1, c2 = st.columns(2)
            with c1:
                cit_count_v = len(res_v.get('citations', []))
                st.markdown(f"""<div class="comp-card">
                    <div class="comp-header vector-header">Vector Search</div>
                    <div class="comp-content">
                        {res_v["answer"]}
                        <div style="margin-top: 15px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                with st.expander(f"📚 Chi tiết nguồn Vector ({cit_count_v})"):
                    render_citations(res_v["citations"], query=msg["compare_data"].get("question", ""))
            with c2:
                cit_count_h = len(res_h.get('citations', []))
                st.markdown(f"""<div class="comp-card">
                    <div class="comp-header hybrid-header">Hybrid Search</div>
                    <div class="comp-content">
                        {res_h["answer"]}
                        <div style="margin-top: 15px;"></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                with st.expander(f"📚 Chi tiết nguồn Hybrid ({cit_count_h})"):
                    render_citations(res_h["citations"], query=msg["compare_data"].get("question", ""))
        else:
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                # Tìm câu hỏi cuối cùng của user ngay trước tin nhắn này để highlight
                user_query = ""
                for m in reversed(st.session_state.messages[:st.session_state.messages.index(msg)]):
                    if m["role"] == "user":
                        user_query = m["content"]
                        break
                render_citations(msg["citations"], query=user_query)

# Chat Input
if prompt := st.chat_input("Nhập câu hỏi tại đây..."):
    if not st.session_state.session_id:
        st.error("Lỗi: Bạn chưa tạo phiên (upload tài liệu).")
    elif not st.session_state.selected_files:
        st.warning("Vui lòng chọn ít nhất một tài liệu nguồn.")
    else:
        current_file_names = list(st.session_state.selected_files)
        st.session_state.messages.append({"role": "user", "content": prompt, "file_names": current_file_names})
        # Render the user message immediately so it's visible while AI thinks
        with st.chat_message("user"):
            live_tags = "".join(
                f'<span style="display:inline-flex;align-items:center;gap:4px;background:#EFF6FF;border:1px solid #BDD7FF;border-radius:20px;padding:2px 9px;font-size:0.72rem;font-weight:600;color:#0056CC;">📄 {fn}</span>'
                for fn in current_file_names
            )
            live_tags_html = f'<div style="display:flex;justify-content:flex-end;flex-wrap:wrap;gap:4px;margin-top:6px;">{live_tags}</div>' if current_file_names else ""
            st.markdown(f'''
                <div style="display: flex; flex-direction: column; align-items: flex-end; width: 100%;">
                    <div style="background-color: #F0F2F6; padding: 12px 18px; border-radius: 20px 20px 4px 20px; color: #202124; text-align: left; display: inline-block; max-width: 85%; font-size: 0.95rem; line-height: 1.5; word-wrap: break-word;">
                        {prompt}
                    </div>
                    {live_tags_html}
                </div>
            ''', unsafe_allow_html=True)
            
        fids = [st.session_state.file_ids_map[f]["id"] for f in st.session_state.selected_files]
        
        with st.chat_message("assistant"):
            if st.session_state.eval_mode:
                with st.spinner("Đang chạy so sánh song song..."):
                    res = API.compare(prompt, st.session_state.session_id, fids, st.session_state.bm25_weight)
                    if res:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "compare_data": {
                                "question": res["question"],
                                "vector": res["vector_result"],
                                "hybrid": res["hybrid_result"]
                            }
                        })
                        st.rerun()
            else:
                with st.spinner("Đang suy nghĩ..."):
                    res = API.ask(
                        prompt, 
                        st.session_state.session_id, 
                        fids, 
                        st.session_state.search_mode, 
                        st.session_state.bm25_weight,
                        st.session_state.rerank_enabled,
                        st.session_state.rerank_threshold
                    )
                    if res:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": res["answer"],
                            "citations": res["citations"],
                            "latency": res.get("latency_ms")
                        })
                        st.rerun()
