import os
import hashlib
import shutil
import fitz 
import re
import pickle
from collections import defaultdict
from tqdm import tqdm
from langchain.schema import Document

# Working directories
companies = {
    "A": {
        "main_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Data\Company A - Aliaxis Corpus",
        "output_pdf_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company A - Aliaxis Corpus\All Unique",
        "output_text_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company A - Aliaxis Corpus\Text Extracted",
        "output_chunks_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company A - Aliaxis Corpus\Text Chunks"
    },
    "B": {
        "main_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Data\Company B - Michelin Corpus",
        "output_pdf_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company B - Michelin Corpus\All Unique",
        "output_text_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company B - Michelin Corpus\Text Extracted",
        "output_chunks_folder": r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data\Company B - Michelin Corpus\Text Chunks"
    }
}

# Create output directories
for company_id, config in companies.items():
    os.makedirs(config["output_pdf_folder"], exist_ok=True)
    os.makedirs(config["output_text_folder"], exist_ok=True)
    os.makedirs(config["output_chunks_folder"], exist_ok=True)

# Function to get file hash for deduplication
def get_file_hash(file_path):

    try:
        with open(file_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except:
        return None

# Function to check if PDF is text-rich or image-heavy
def is_text_rich_pdf(file_path, min_chars_per_page=100):

    try:
        with fitz.open(file_path) as doc:
            total_chars = 0
            total_pages = len(doc)
            
            if total_pages == 0:
                return False
                
            for page in doc:
                text = page.get_text("text").strip()
                total_chars += len(text)
            
            avg_chars_per_page = total_chars / total_pages
            return avg_chars_per_page >= min_chars_per_page and total_chars >= 500
            
    except:
        return False  # Consider failed analysis as not text-rich

# Function to normalize unicode punctuation
def normalize_unicode_punctuation(text):
    """Fix common unicode issues."""
    return (text.replace(""", '"').replace(""", '"')
                .replace("'", "'").replace("'", "'")
                .replace("–", "-").replace("—", "-")
                .replace("•", "-"))

# Function to flag headings
def is_heading(text: str) -> bool:
    text = text.strip()

    # Early exits
    if not text or len(text.split()) > 10:
        return False

    # Must contain at least one alphabetic character
    if not re.search(r'[A-Za-z]', text):
        return False

    ends_with_colon = text.endswith(":")
    is_title_case = text == text.title()
    is_all_caps = text.isupper()
    lacks_terminal_punctuation = not text.endswith(('.', '?', '!'))

    # Short phrase with colon
    if ends_with_colon and len(text.split()) <= 6:
        return True

    # Title or all-caps style, not a sentence
    return (is_title_case or is_all_caps) and lacks_terminal_punctuation

# Function to check if a block of text is likely a table
def is_table_block(text):
    lines = text.splitlines()
    if len(lines) < 2:
        return False
    
    # Count numbers and spacing patterns
    numeric_count = sum(len(re.findall(r'\d+', line)) for line in lines)
    tab_count = sum(line.count('\t') for line in lines)
    multi_space_count = sum(len(re.findall(r'\s{2,}', line)) for line in lines)
    
    return numeric_count >= 3 and (tab_count > 0 or multi_space_count > len(lines))

# Function to clean text for embeddings
def clean_text_for_embeddings(text):
    """Clean text for processing."""
    text = re.sub(r'[\b\x08]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)  # fix hyphen breaks

    # Remove URLs, links, domains, and email addresses
    text = re.sub(r'https?://[^\s]+', '', text)  
    text = re.sub(r'www\.[^\s]+', '', text)      
    text = re.sub(r'[^\s]+\.(com|org|net|edu|gov|co\.uk|de|fr)\b[^\s]*', '', text)  
    text = re.sub(r'\S+@\S+\.\S+', '', text)     

    # Normalize punctuation using previous function, lowercase, and trim
    text = normalize_unicode_punctuation(text)
    text = text.lower()  
    return text.strip()

# Function to estimate token count based on character length
def estimate_tokens(text):

    # Rough estimate, given that 1 token ≈ 4 characters
    return len(text) // 4

# Function to chunk text by sections and size with overlap of 50 characters
def chunk_text_by_sections(text, chunk_size=500, overlap=50):

    chunks = []
    
    # Split by section headers first
    sections = re.split(r'(\n### .+ ###\n)', text)
    
    current_chunk = ""
    current_size = 0
    
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        section_size = estimate_tokens(section)
        
        # If section itself is too large, split it further
        if section_size > chunk_size:

            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""
                current_size = 0
            
            # Split large section
            sentences = re.split(r'(?<=[.!?])\s+', section)
            temp_chunk = ""
            temp_size = 0
            
            for sentence in sentences:
                sentence_size = estimate_tokens(sentence)
                
                if temp_size + sentence_size > chunk_size and temp_chunk:

                    # Add overlap from previous chunk if exists
                    if chunks:
                        overlap_text = chunks[-1].split()[-overlap//4:]  # Rough word-based overlap, according to token estimate
                        temp_chunk = " ".join(overlap_text) + " " + temp_chunk
                    
                    chunks.append(temp_chunk.strip())
                    
                    # Start new chunk with overlap
                    overlap_words = temp_chunk.split()[-overlap//4:]
                    temp_chunk = " ".join(overlap_words) + " " + sentence
                    temp_size = estimate_tokens(temp_chunk)
                else:
                    temp_chunk += " " + sentence if temp_chunk else sentence
                    temp_size += sentence_size
            
            # Add remaining content
            if temp_chunk.strip():
                chunks.append(temp_chunk.strip())
        
        # If adding this section would exceed chunk size
        elif current_size + section_size > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap
            if chunks:
                overlap_words = current_chunk.split()[-overlap//4:]
                current_chunk = " ".join(overlap_words) + " " + section
            else:
                current_chunk = section
            current_size = estimate_tokens(current_chunk)
        else:
            current_chunk += "\n" + section if current_chunk else section
            current_size += section_size
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# Main processing loop
all_documents = []

# Step 1: Organize PDF files
for company_id, config in companies.items():
    
    main_folder = config["main_folder"]
    output_pdf_folder = config["output_pdf_folder"]
    output_text_folder = config["output_text_folder"]
    output_chunks_folder = config["output_chunks_folder"]

    # Create folder for non-text-heavy PDFs for checking
    rejected_pdf_folder = os.path.join(os.path.dirname(output_pdf_folder), f"{company_id}_non_text_heavy_pdfs")
    os.makedirs(rejected_pdf_folder, exist_ok=True)

    # Obtain list of all PDF files
    pdf_files = []
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(subfolder_path, filename))

    # Number of pdf files found
    print(f"Found {len(pdf_files)} PDF files")

    # Step 2: Deduplicate and filter text-rich PDFs
    seen_hashes = set()
    file_counter = 1 # Starts from 1 because 0 is not ideal for filenames
    duplicate_counter = 0
    error_counter = 0
    image_heavy_counter = 0

    for file_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:

            # Check file hash for deduplication
            file_hash = get_file_hash(file_path)
            if file_hash is None:
                error_counter += 1
                continue

            # If file hash is already present in the list, add to duplicate count 
            if file_hash in seen_hashes:
                duplicate_counter += 1
                continue
            
            seen_hashes.add(file_hash)
            
            # Check if PDF is text-rich
            if not is_text_rich_pdf(file_path):

                # If not, it is moved to the folder for rejected PDFs
                filename = os.path.basename(file_path)
                rejected_path = os.path.join(rejected_pdf_folder, filename)
                shutil.copyfile(file_path, rejected_path)
                image_heavy_counter += 1
                continue
            
            # Copy to output folder
            new_filename = f"{company_id}_{file_counter}.pdf" # Use the file counter for naming
            new_file_path = os.path.join(output_pdf_folder, new_filename)
            shutil.copyfile(file_path, new_file_path)

            # Update file counter after deduplication and text-rich check
            file_counter += 1

        # If an error occurs in this process, increment the error counter    
        except:
            error_counter += 1

    print(f"Text-rich PDFs copied: {file_counter - 1}") # -1 because counter starts at 1
    print(f"Duplicates skipped: {duplicate_counter}")
    print(f"Image-heavy PDFs moved to separate folder: {image_heavy_counter}")
    print(f"Errors: {error_counter}")

    # Step 3: Extract text from PDFs
    pdf_paths = [os.path.join(output_pdf_folder, f) for f in os.listdir(output_pdf_folder) 
                if f.lower().endswith(".pdf")]
    processed = 0
    text_errors = 0

    for pdf_path in tqdm(pdf_paths, desc="Extracting"):
        try:
            filename = os.path.basename(pdf_path)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_text_folder, output_filename)

            with fitz.open(pdf_path) as doc:
                full_text = ""

                # Extract text page by page and handle columns of text
                for page in doc:
                    blocks = page.get_text("blocks")
                    page_width = page.rect.width

                    left_column = []
                    right_column = []

                    for b in blocks:
                        if len(b) < 5:
                            continue
                        x0, y0, x1, y1, block_text = b[:5]
                        if not block_text.strip():
                            continue

                        block_text = block_text.replace("\n", " ").strip()

                        # Skip table blocks
                        if is_table_block(block_text):
                            continue 

                        # Assign block to left or right column based on x0 position
                        if x0 < page_width / 2:  
                            left_column.append((y0, block_text))   # Left column
                        else:
                            right_column.append((y0, block_text))  # Right column

                    # Sort blocks by vertical position and then merge columns
                    left_column.sort()
                    right_column.sort()
                    ordered_blocks = [b[1] for b in left_column + right_column]

                    # Merge blocks into lines
                    page_lines = []
                    for block in ordered_blocks:
                        lines = block.strip().splitlines()
                        for line in lines:
                            line = line.strip()
                            if not line:  # Skip empty lines
                                continue

                            # Flag headings
                            if is_heading(line):
                                page_lines.append(f"\n### {line.strip()} ###\n")
                            else:
                                page_lines.append(line.strip())

                    full_text += "\n\n".join(page_lines) + "\n"

                # Final text cleaning for embeddings
                cleaned = clean_text_for_embeddings(full_text)

            if cleaned and len(cleaned) > 50:  # Only save if meaningful content
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(cleaned)
                processed += 1
            else:
                text_errors += 1

        except:
            text_errors += 1

    print(f"Text files created: {processed}")
    print(f"Text extraction errors: {text_errors}")

    # Step 4: Chunk text files
    text_files = [f for f in os.listdir(output_text_folder) if f.endswith('.txt')]
    chunk_counter = 0
    chunk_errors = 0

    for text_file in tqdm(text_files, desc="Chunking"):
        try:
            input_path = os.path.join(output_text_folder, text_file)

            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                chunk_errors += 1
                continue
            
            # Chunk the text
            chunks = chunk_text_by_sections(text, chunk_size=500, overlap=50)
            
            if not chunks:
                chunk_errors += 1
                continue
            
            # Save chunks
            base_name = os.path.splitext(text_file)[0]
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{base_name}_chunk_{i+1:03d}.txt"
                chunk_path = os.path.join(output_chunks_folder, chunk_filename)

                with open(chunk_path, 'w', encoding='utf-8') as f:

                    # Add metadata in header of chunk file
                    f.write(f"SOURCE: {base_name}\n")
                    f.write(f"COMPANY: {company_id}\n")
                    f.write(f"CHUNK: {i+1}/{len(chunks)}\n")
                    f.write(f"TOKENS: ~{estimate_tokens(chunk)}\n") #token count based on estimate function
                    f.write("---\n")
                    f.write(chunk)
                
                chunk_counter += 1
        
        except Exception as e:
            chunk_errors += 1

    print(f"Total chunks created: {chunk_counter}")
    print(f"Chunking errors: {chunk_errors}")

    # Step 5: Create Document objects with metadata previously saved in the chunk files
    company_documents = []
    metadata_errors = 0

    for chunk_file in tqdm(os.listdir(output_chunks_folder), desc="Loading chunks"):
        if not chunk_file.endswith(".txt"):
            continue

        path = os.path.join(output_chunks_folder, chunk_file)

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Initialize metadata with filename and company info
            metadata = {
                "chunk_file": chunk_file,
                "file_path": path,
                "company_id": company_id
            }
            
            # Check if content has metadata header
            if '---\n' in content:
                parts = content.split('---\n', 1)
                metadata_section = parts[0].strip()
                text = parts[1].strip() if len(parts) > 1 else ""
                
                # Parse metadata lines
                for line in metadata_section.splitlines():
                    line = line.strip()
                    if ": " in line:
                        key, value = line.split(": ", 1)
                        metadata[key.lower().strip()] = value.strip()
            else:
                # No metadata separator found, treat entire content as text
                text = content.strip()
                metadata_errors += 1
                
            # Add additional computed metadata
            metadata["chunk_length"] = len(text)
            
            # Create Document object
            doc = Document(page_content=text, metadata=metadata)
            company_documents.append(doc)
            
        except Exception as e:
            print(f"Error processing {chunk_file}: {e}")
            metadata_errors += 1

    print(f"{len(company_documents)} documents for Company {company_id}")
    print(f"Metadata parsing errors: {metadata_errors}")
    
    # Add company documents to the master list
    all_documents.extend(company_documents)

# Save all documents to a pickle file for later use
output_base = r"C:\Users\jtgb0\OneDrive\Ambiente de Trabalho\Output Data"
pkl_path = os.path.join(output_base, "all_documents.pkl")
with open(pkl_path, "wb") as f:
    pickle.dump(all_documents, f)
