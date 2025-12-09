import os
import fitz  
from flask import Flask, render_template, request, flash, session, jsonify
from flask_session import Session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from youtube_transcript_api import YouTubeTranscriptApi, CouldNotRetrieveTranscript
from urllib.parse import urlparse, parse_qs


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
user_agent = os.getenv("USER_AGENT", "MyApp/1.0")


app2 = Flask(__name__)
app2.secret_key = "supersecretkey"
app2.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app2.config["UPLOAD_FOLDER"], exist_ok=True)

app2.config["SESSION_TYPE"] = "filesystem"
app2.config["SESSION_FILE_DIR"] = os.path.join(os.getcwd(), "flask_session")
app2.config["SESSION_PERMANENT"] = False
app2.config["SESSION_USE_SIGNER"] = True
os.makedirs(app2.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app2)

MAX_CONTEXT_CHARS = 20000  

def get_youtube_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        qs = parse_qs(parsed.query)
        return qs.get("v", [None])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    return None

@app2.route("/", methods=["GET", "POST"])
def index():
    summary = None
    pdf_length = None

    if "history" not in session:
        session["history"] = []

    if request.method == "POST":
        generic_url = request.form.get("generic_url", "").strip()
        uploaded_pdf = request.files.get("uploaded_pdf")
        model_choice = request.form.get("model_choice", "llama-3.1-8b-instant")
        lang_pref = request.form.get("lang_pref", "en").strip()
        summary_length = request.form.get("summary_length", "medium")

        word_limit = {"short": 100, "medium": 300, "long": 600}.get(summary_length, 300)

        if not groq_api_key:
            flash("❌ GROQ_API_KEY not set in .env file", "error")
            return render_template("index.html")

        if not generic_url and not uploaded_pdf:
            flash("❌ Please enter a URL or upload a PDF.", "error")
            return render_template("index.html")

        try:
            llm = ChatGroq(model=model_choice, groq_api_key=groq_api_key)
            docs = []


            if "youtube.com" in generic_url or "youtu.be" in generic_url:
                video_id = get_youtube_video_id(generic_url)
                if not video_id:
                    flash("❌ Could not extract video ID from YouTube URL.", "error")
                else:
                    try:
                        yt_api = YouTubeTranscriptApi()
                        transcript_list = yt_api.fetch(
                            video_id,
                            languages=[lang.strip() for lang in lang_pref.split(",")]
                        )
                        transcript_text = " ".join([entry.text for entry in transcript_list])
                        if len(transcript_text) > 30000:
                            transcript_text = transcript_text[:30000]
                        docs = [Document(page_content=transcript_text, metadata={"source": generic_url})]
                    except CouldNotRetrieveTranscript:
                        flash("⚠️ Transcript not available in the specified language(s).", "warning")
                    except Exception as e:
                        flash(f"❌ Failed to fetch transcript: {e}", "error")



            elif generic_url:
                loader = WebBaseLoader(
                    web_paths=[generic_url],
                    requests_kwargs={"headers": {"User-Agent": user_agent}},
                )
                docs = loader.load()

            elif uploaded_pdf:
                filename = secure_filename(uploaded_pdf.filename)
                pdf_path = os.path.join(app2.config["UPLOAD_FOLDER"], filename)
                uploaded_pdf.save(pdf_path)
                with fitz.open(pdf_path) as pdf_doc:
                    pdf_length = pdf_doc.page_count
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                os.remove(pdf_path)

            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)

                map_prompt = PromptTemplate(
                    template="Summarize this section briefly in {lang}:\n{text}",
                    input_variables=["text", "lang"],
                )
                combine_prompt = PromptTemplate(
                    template="Combine these summaries into one coherent summary in {lang}, about {word_limit} words:\n{text}",
                    input_variables=["text", "lang", "word_limit"],
                )

                chain = load_summarize_chain(
                    llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                )

                output = chain.invoke({
                    "input_documents": split_docs,
                    "lang": lang_pref,
                    "word_limit": word_limit
                })

                summary = output.get("output_text", str(output))
                flash("✅ Summary generated successfully!", "success")

                session["history"].append({
                    "url": generic_url or f"Uploaded PDF ({pdf_length or 'Unknown'} pages)",
                    "summary": summary
                })
                session["history"] = session["history"][-5:]
                session.modified = True

        except Exception as e:
            flash(f"❌ Error: {e}", "error")

    return render_template("index.html", summary=summary, pdf_length=pdf_length, history=session["history"])


@app2.route("/pdf_parser", methods=["GET"])
def pdf_parser_page():
    parsed = bool(session.get("parsed_text"))
    pdf_length = session.get("pdf_length")
    return render_template("pdf_parser.html", parsed=parsed, pdf_length=pdf_length)


@app2.route("/pdf_parser/upload", methods=["POST"])
def pdf_parser_upload():
    uploaded_pdf = request.files.get("uploaded_pdf")
    if not uploaded_pdf or uploaded_pdf.filename == "":
        return jsonify({"ok": False, "error": "No PDF uploaded"}), 400

    filename = secure_filename(uploaded_pdf.filename)
    pdf_path = os.path.join(app2.config["UPLOAD_FOLDER"], filename)
    uploaded_pdf.save(pdf_path)

    try:
        with fitz.open(pdf_path) as pdf_doc:
            pdf_length = pdf_doc.page_count

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        parsed_text = "\n\n".join([d.page_content for d in split_docs])

        session["parsed_text"] = parsed_text
        session["pdf_length"] = pdf_length
        session.modified = True  

        return jsonify({"ok": True, "pdf_length": pdf_length})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            pass


@app2.route("/pdf_parser/ask", methods=["POST"])
def pdf_parser_ask():
    data = request.get_json()
    question = data.get("question")
    model_choice = data.get("model", "llama-3.1-8b-instant")

    parsed_text = session.get("parsed_text")
    if not parsed_text:
        return jsonify({"ok": False, "error": "No PDF parsed yet"}), 400

    if not question:
        return jsonify({"ok": False, "error": "No question provided"}), 400

    try:
        llm = ChatGroq(model=model_choice, groq_api_key=groq_api_key)
        qa_prompt = PromptTemplate(
            template=(
                "Use the PDF content below to answer.\n\n"
                "PDF Content:\n{text}\n\n"
                "Question: {question}\n\n"
                "Answer only from the document."
            ),
            input_variables=["text", "question"]
        )

        chain = LLMChain(prompt=qa_prompt, llm=llm)
        result = chain.invoke({"text": parsed_text[:MAX_CONTEXT_CHARS], "question": question})

        answer = result.get("text") if isinstance(result, dict) else str(result)
        return jsonify({"ok": True, "answer": answer})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app2.route("/pdf_parser/clear", methods=["POST"])
def pdf_parser_clear():
    session.pop("parsed_text", None)
    session.pop("pdf_length", None)
    session.modified = True
    return jsonify({"ok": True})


@app2.route("/debug/session")
def debug_session():
    return jsonify({
        "keys": list(session.keys()),
        "has_parsed_text": bool(session.get("parsed_text")),
        "pdf_length": session.get("pdf_length")
    })


if __name__ == "__main__":
    app2.run(debug=True, use_reloader=False)
