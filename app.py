import os
import json
import base64
import shutil
import zipfile
import re # Import regex module
import subprocess
from pathlib import Path
from uuid import uuid4
from flask import Flask, request, render_template, jsonify, send_from_directory, url_for
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import PyPDF2
import tempfile

load_dotenv()

print("--- .env Loading Debug ---")
dotenv_api_key = os.getenv("MISTRAL_API_KEY")
if dotenv_api_key:
    print(f"API Key loaded from .env (first 4 chars): {dotenv_api_key[:4]}...")
else:
    print("API Key NOT loaded from .env. Check .env file and setup.")
print("--- End .env Debug ---")


app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = Path(os.getenv('UPLOAD_FOLDER', 'uploads'))
OUTPUT_FOLDER = Path(os.getenv('OUTPUT_FOLDER', 'output'))
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
ALLOWED_EXTENSIONS = {'pdf'}
PAGE_SEPARATOR_DEFAULT = os.getenv('PAGE_SEPARATOR', '---')

UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def replace_images_in_markdown_with_wikilinks(markdown_str: str, image_mapping: dict, use_wikilinks: bool = True) -> str:
    """
    替换 markdown 中的图片链接
    
    Args:
        markdown_str: 原始 markdown 字符串
        image_mapping: 图片映射字典 {original_id: new_name}
        use_wikilinks: 是否使用 Obsidian wikilinks 格式 (True) 或标准 markdown 格式 (False)
    
    Returns:
        更新后的 markdown 字符串
    """
    updated_markdown = markdown_str
    for original_id, new_name in image_mapping.items():
        if use_wikilinks:
            # Obsidian wikilinks 格式: ![[image-name]]
            updated_markdown = updated_markdown.replace(
                f"![{original_id}]({original_id})",
                f"![[{new_name}]]"
            )
        else:
            # 标准 markdown 格式: ![alt](path/to/image)
            updated_markdown = updated_markdown.replace(
                f"![{original_id}]({original_id})",
                f"![{new_name}](images/{new_name})"
            )
    return updated_markdown

# --- Core Processing Logic ---

def process_pdf(pdf_path: Path, api_key: str, session_output_dir: Path, page_separator: str | None = PAGE_SEPARATOR_DEFAULT, use_wikilinks: bool = False) -> tuple[str, str, list[str], Path, Path]:
    """
    Processes a single PDF file using Mistral OCR and saves results.

    Args:
        pdf_path: Path to the PDF file.
        api_key: Mistral API key.
        session_output_dir: Directory to store output.
        page_separator: Text to insert between pages. Use empty string to join pages without a separator.

    Returns:
        A tuple (pdf_base_name, final_markdown_content, list_of_image_filenames, path_to_markdown_file, path_to_images_dir)
    Raises:
        Exception: For processing errors.
    """
    pdf_base = pdf_path.stem
    base_sanitized_original = secure_filename(pdf_base)
    pdf_base_sanitized = base_sanitized_original
    print(f"Processing {pdf_path.name}...")

    pdf_output_dir = session_output_dir / pdf_base_sanitized
    counter = 1
    while pdf_output_dir.exists():
        pdf_base_sanitized = f"{base_sanitized_original}_{counter}"
        pdf_output_dir = session_output_dir / pdf_base_sanitized
        counter += 1

    pdf_output_dir.mkdir(exist_ok=True)
    images_dir = pdf_output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 分割PDF文件
    split_files = split_pdf(pdf_path)
    if len(split_files) > 1:
        print(f"PDF has been split into {len(split_files)} parts for processing")

    client = Mistral(api_key=api_key)
    all_markdown_pages = []
    all_image_filenames = []
    global_image_counter = 1
    uploaded_files = []

    try:
        for split_index, split_file in enumerate(split_files):
            print(f"Processing split {split_index + 1}/{len(split_files)}...")
            
            # 上传文件到Mistral
            with open(split_file, "rb") as f:
                pdf_bytes = f.read()
            uploaded_file = client.files.upload(
                file={"file_name": f"{pdf_path.name}_part_{split_index+1}", "content": pdf_bytes}, 
                purpose="ocr"
            )
            uploaded_files.append(uploaded_file)

            print(f"  File uploaded (ID: {uploaded_file.id}). Getting signed URL...")
            signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=60)

            print(f"  Calling OCR API...")
            ocr_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=True
            )
            print(f"  OCR processing complete for split {split_index + 1}.")

            # 处理OCR响应
            for page_index, page in enumerate(ocr_response.pages):
                current_page_markdown = page.markdown
                page_image_mapping = {}

                for image_obj in page.images:
                    base64_str = image_obj.image_base64
                    if not base64_str: continue

                    if base64_str.startswith("data:"):
                        try: base64_str = base64_str.split(",", 1)[1]
                        except IndexError: continue

                    try: image_bytes = base64.b64decode(base64_str)
                    except Exception as decode_err:
                        print(f"  Warning: Base64 decode error for image {image_obj.id} on page {page_index+1}: {decode_err}")
                        continue

                    original_ext = Path(image_obj.id).suffix
                    ext = original_ext if original_ext else ".png"
                    new_image_name = f"{pdf_base_sanitized}_p{len(all_markdown_pages)+1}_img{global_image_counter}{ext}"
                    global_image_counter += 1

                    image_output_path = images_dir / new_image_name
                    try:
                        with open(image_output_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        all_image_filenames.append(new_image_name)
                        page_image_mapping[image_obj.id] = new_image_name
                    except IOError as io_err:
                        print(f"  Warning: Could not write image file {image_output_path}: {io_err}")
                        continue

                updated_page_markdown = replace_images_in_markdown_with_wikilinks(current_page_markdown, page_image_mapping, use_wikilinks=use_wikilinks)
                all_markdown_pages.append(updated_page_markdown)

            # 清理Mistral文件
            try:
                client.files.delete(file_id=uploaded_file.id)
                print(f"  Deleted temporary file {uploaded_file.id} from Mistral.")
            except Exception as delete_err:
                print(f"  Warning: Could not delete file {uploaded_file.id} from Mistral: {delete_err}")

        # 合并所有markdown内容
        if page_separator:
            # 如果用户指定了页面分隔符，使用分隔符连接
            separator = f"\n\n{page_separator}\n\n"
            final_markdown_content = separator.join(all_markdown_pages)
        else:
            # 如果没有指定分隔符，直接连接，避免错误分割段落
            final_markdown_content = " ".join(all_markdown_pages)
        output_markdown_path = pdf_output_dir / f"{pdf_base_sanitized}_output.md"

        try:
            with open(output_markdown_path, "w", encoding="utf-8") as md_file:
                md_file.write(final_markdown_content)
            print(f"  Markdown generated successfully at {output_markdown_path}")
        except IOError as io_err:
            raise Exception(f"Failed to write final markdown file: {io_err}") from io_err

        return pdf_base_sanitized, final_markdown_content, all_image_filenames, output_markdown_path, images_dir

    except Exception as e:
        error_str = str(e)
        # Attempt to extract JSON error message from the exception string
        json_index = error_str.find('{')
        if json_index != -1:
            try:
                error_json = json.loads(error_str[json_index:])
                error_msg = error_json.get("message", error_str)
            except Exception:
                error_msg = error_str
        else:
            error_msg = error_str
        print(f"  Error processing {pdf_path.name}: {error_msg}")
        
        # 清理所有上传的文件
        for uploaded_file in uploaded_files:
            try:
                client.files.delete(file_id=uploaded_file.id)
            except Exception:
                pass
        raise Exception(error_msg)
    finally:
        # 清理临时分割文件
        for split_file in split_files:
            if split_file != pdf_path:  # 不要删除原始文件
                try:
                    split_file.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete temporary split file {split_file}: {e}")


def create_zip_archive(source_dir: Path, output_zip_path: Path):
    print(f"  Creating ZIP archive: {output_zip_path} from {source_dir}")
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for entry in source_dir.rglob('*'):
                arcname = entry.relative_to(source_dir)
                # Ensure arcname contains the 'images' folder if it exists
                # Example: arcname should be 'images/my_image.png', not just 'my_image.png'
                zipf.write(entry, arcname)
                print(f"    Added to ZIP: {arcname}")
        print(f"  Successfully created ZIP: {output_zip_path}")
    except Exception as e:
        print(f"  Error creating ZIP file {output_zip_path}: {e}")
        raise


def create_pdf_from_markdown(markdown_path: Path, output_pdf_path: Path) -> bool:
    """
    使用 md-to-pdf 将 markdown 文件转换为 PDF
    
    Args:
        markdown_path: markdown 文件路径
        output_pdf_path: 输出 PDF 文件路径（预期路径）
        
    Returns:
        bool: 转换是否成功
    """
    try:
        print(f"  Converting markdown to PDF: {markdown_path}")
        
        # 检查 md-to-pdf 是否安装
        result = subprocess.run(['md-to-pdf', '--version'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("  Warning: md-to-pdf not found. Install with: npm install -g md-to-pdf")
            return False
        
        # 执行转换
        # 获取 CSS 文件的绝对路径
        css_path = Path(__file__).parent / 'md2pdf.css'
        cmd = [
            'md-to-pdf', 
            str(markdown_path),
            '--pdf-options', '{"outline": true, "format":"A4"}',
            '--stylesheet', str(css_path),
            '--launch-options', '{"args": ["--no-sandbox", "--disable-setuid-sandbox"]}'
        ]
        
        print(f"  Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # md-to-pdf 会自动生成与输入文件同名的 PDF 文件
            expected_pdf_path = markdown_path.with_suffix('.pdf')
            if expected_pdf_path.exists():
                print(f"  Successfully created PDF: {expected_pdf_path}")
                return True
            else:
                print(f"  Error: Expected PDF file not found: {expected_pdf_path}")
                return False
        else:
            error_msg = result.stderr
            if "Could not find Chrome" in error_msg:
                print(f"  Error: Chrome browser not found. Please install Chrome or run: npx puppeteer browsers install chrome")
            else:
                print(f"  Error creating PDF: {error_msg}")
            return False
            
    except Exception as e:
        print(f"  Error in PDF conversion: {e}")
        return False


def split_pdf(pdf_path: Path, max_pages: int = 1000) -> list[Path]:
    """
    将PDF文件分割成多个不超过max_pages页的文件
    
    Args:
        pdf_path: PDF文件路径
        max_pages: 每个分割文件的最大页数
        
    Returns:
        包含所有分割文件路径的列表
    """
    temp_files = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            if total_pages <= max_pages:
                return [pdf_path]
            
            # 计算需要分割的文件数
            num_splits = (total_pages + max_pages - 1) // max_pages
            
            for i in range(num_splits):
                start_page = i * max_pages
                end_page = min((i + 1) * max_pages, total_pages)
                
                # 创建新的PDF写入器
                pdf_writer = PyPDF2.PdfWriter()
                
                # 添加页面
                for page_num in range(start_page, end_page):
                    pdf_writer.add_page(pdf_reader.pages[page_num])
                
                # 创建临时文件
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                temp_files.append(Path(temp_file.name))
                
                # 写入分割后的PDF
                with open(temp_file.name, 'wb') as output_file:
                    pdf_writer.write(output_file)
                
                print(f"Created split PDF {i+1}/{num_splits} with pages {start_page+1}-{end_page}")
            
            return temp_files
            
    except Exception as e:
        # 清理临时文件
        for temp_file in temp_files:
            try:
                temp_file.unlink()
            except:
                pass
        raise Exception(f"Error splitting PDF: {str(e)}")


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html', default_page_separator='')

@app.route('/check-api-key', methods=['GET'])
def check_api_key():
    """check if the API key is configured in the environment variables"""
    api_key = os.getenv("MISTRAL_API_KEY")
    return jsonify({"has_api_key": bool(api_key)})

@app.route('/process', methods=['POST'])
def handle_process():
    if 'pdf_files' not in request.files:
        return jsonify({"error": "No PDF files part in the request"}), 400

    files = request.files.getlist('pdf_files')
    api_key = request.form.get('api_key')
    output_format = request.form.get('output_format', 'standard')  # 默认为标准格式
    generate_pdf = request.form.get('generate_pdf') == 'true'  # 是否生成 PDF 版本
    
    print(f"Form data received:")
    print(f"  - output_format: {output_format}")
    print(f"  - generate_pdf: {generate_pdf}")
    print(f"  - files count: {len(files)}")

    if not api_key:
        print("API Key from web form is empty.")
        # Check if we have a fallback API key from .env (or elsewhere - server-side config)
        api_key_fallback = os.getenv("MISTRAL_API_KEY") # Try to get from env again
        if api_key_fallback:
            api_key = api_key_fallback
            print(f"Using fallback API Key from environment (first 4 chars): {api_key[:4]}...") # Debug print
        else:
            return jsonify({"error": "Mistral API Key is required"}), 400

    if not files or all(f.filename == '' for f in files):
         return jsonify({"error": "No selected PDF files"}), 400

    valid_files, invalid_files = [], []
    for f in files:
        if f and allowed_file(f.filename): valid_files.append(f)
        elif f and f.filename != '': invalid_files.append(f.filename)

    if not valid_files:
         # ... (error handling as before) ...
         error_msg = "No valid PDF files found."
         if invalid_files: error_msg += f" Invalid files skipped: {', '.join(invalid_files)}"
         return jsonify({"error": error_msg}), 400


    session_id = str(uuid4())
    session_upload_dir = UPLOAD_FOLDER / session_id
    session_output_dir = OUTPUT_FOLDER / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    session_output_dir.mkdir(parents=True, exist_ok=True)

    processed_files_results = [] # Changed name for clarity
    processing_errors = []
    if invalid_files: processing_errors.append(f"Skipped non-PDF files: {', '.join(invalid_files)}")

    page_separator = request.form.get('page_separator')
    if page_separator is None or page_separator.strip() == '':
        page_separator = None  # 使用 None 表示不使用分隔符
    else:
        page_separator = page_separator.strip()  # 去除空白字符

    for file in valid_files:
        original_filename = file.filename
        filename_sanitized = secure_filename(original_filename)
        temp_pdf_path = session_upload_dir / filename_sanitized


        try:
            print(f"Saving uploaded file temporarily to: {temp_pdf_path}")
            file.save(temp_pdf_path)

            # Process PDF - Capture new return values
            use_wikilinks = (output_format == 'obsidian')
            processed_pdf_base, markdown_content, image_filenames, md_path, img_dir = process_pdf(
                temp_pdf_path, api_key, session_output_dir, page_separator, use_wikilinks=use_wikilinks
            )

            # Generate PDF version if requested (before creating ZIP)
            pdf_filename = None
            pdf_download_url = None
            if generate_pdf:
                # md-to-pdf 会自动生成与输入文件同名的 PDF 文件
                if create_pdf_from_markdown(md_path, None):  # 不需要指定输出路径
                    pdf_filename = f"{processed_pdf_base}_output.pdf"
                    pdf_download_url = url_for('download_file', session_id=session_id, filename=pdf_filename, _external=True)
                    print(f"  PDF generated successfully: {pdf_filename}")
                else:
                    print(f"  Warning: Failed to generate PDF")

            # Create ZIP (using the individual output dir) - now includes PDF if generated
            zip_filename = f"{processed_pdf_base}_output.zip"
            zip_output_path = session_output_dir / zip_filename
            individual_output_dir = session_output_dir / processed_pdf_base
            create_zip_archive(individual_output_dir, zip_output_path)

            download_url = url_for('download_file', session_id=session_id, filename=zip_filename, _external=True)

            # Store results including preview data
            result_item = {
                "original_filename": original_filename, # Keep original name for display
                "zip_filename": zip_filename,
                "download_url": download_url,
                "preview": {
                    "markdown": markdown_content,
                    "images": image_filenames,
                    "pdf_base": processed_pdf_base # Use the sanitized base name returned by process_pdf
                }
            }
            
            # Add PDF information if generated
            if generate_pdf and pdf_filename and pdf_download_url:
                result_item["pdf_filename"] = pdf_filename
                result_item["pdf_download_url"] = pdf_download_url
            
            processed_files_results.append(result_item)
            print(f"Successfully processed and zipped: {original_filename}")

        except Exception as e:
            print(f"ERROR: Failed processing {original_filename}: {e}")
            processing_errors.append(f"{original_filename}: Processing Error - {e}")
        finally:
            if temp_pdf_path.exists():
                try: temp_pdf_path.unlink()
                except OSError as unlink_err: print(f"Warning: Could not delete temp file {temp_pdf_path}: {unlink_err}")

    # Cleanup session upload dir
    try:
        shutil.rmtree(session_upload_dir)
        print(f"Cleaned up session upload directory: {session_upload_dir}")
    except OSError as rmtree_err:
        print(f"Warning: Could not delete session upload directory {session_upload_dir}: {rmtree_err}")

    if not processed_files_results and processing_errors:
         return jsonify({"error": "All PDF processing attempts failed.", "details": processing_errors}), 500
    elif not processed_files_results:
         return jsonify({"error": "No files were processed successfully."}), 500
    else:
        # Return session_id along with results for constructing image URLs on frontend
        return jsonify({
            "success": True,
            "session_id": session_id, # ADDED session_id here
            "results": processed_files_results, # Renamed from 'downloads'
            "errors": processing_errors
        }), 200

# --- NEW ROUTE for serving images ---
@app.route('/view_image/<session_id>/<pdf_base>/<filename>')
def view_image(session_id, pdf_base, filename):
    """Serves an extracted image file for inline display."""
    safe_session_id = secure_filename(session_id)
    safe_pdf_base = secure_filename(pdf_base)
    safe_filename = secure_filename(filename)

    # Construct path relative to the *pdf_base* specific output folder
    directory = OUTPUT_FOLDER / safe_session_id / safe_pdf_base / "images"
    file_path = directory / safe_filename

    # Security check
    if not str(file_path.resolve()).startswith(str(directory.resolve())):
         return "Invalid path", 400
    if not file_path.is_file():
         return "Image not found", 404

    print(f"Serving image: {file_path}")
    # Send *without* as_attachment=True for inline display
    return send_from_directory(directory, safe_filename)


@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    """Serves the generated ZIP file for download."""
    safe_session_id = secure_filename(session_id)
    safe_filename = secure_filename(filename)
    directory = OUTPUT_FOLDER / safe_session_id
    file_path = directory / safe_filename

    if not str(file_path.resolve()).startswith(str(directory.resolve())): return "Invalid path", 400
    if not file_path.is_file(): return "File not found", 404

    print(f"Serving ZIP for download: {file_path}")
    return send_from_directory(directory, safe_filename, as_attachment=True)

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(host=host, port=port, debug=debug_mode)
