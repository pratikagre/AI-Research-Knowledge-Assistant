from typing import List, Dict, Any

class Chunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using separators.
        """
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            # No separators left, force split at chunk_size
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator = separators[0]
        splits = []
        
        # If separator is empty string, split character by character
        if separator == "":
            parts = list(text)
        else:
            parts = text.split(separator)

        current_doc = []
        for part in parts:
            # Re-add separator if it's not empty string and not the last element
            actual_part = part
            if separator != "" and separator != " " and separator != "\n" and separator != "\n\n":
                actual_part = part + separator

            if len(actual_part) > self.chunk_size:
                # If a single part exceeds chunk size, split it with remaining separators
                if current_doc:
                    splits.append(separator.join(current_doc))
                    current_doc = []
                splits.extend(self._split_text(actual_part, separators[1:]))
            else:
                current_len = sum(len(p) for p in current_doc) + (len(separator) * (len(current_doc) - 1) if current_doc else 0)
                if current_len + len(actual_part) > self.chunk_size:
                    if current_doc:
                        splits.append(separator.join(current_doc))
                    current_doc = [actual_part]
                else:
                    current_doc.append(actual_part)

        if current_doc:
            splits.append(separator.join(current_doc))

        return splits

    def chunk_document(self, doc_id: str, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits page text into overlapping chunks while preserving page number annotations.
        """
        chunks = []
        chunk_id = 0

        for page in pages_data:
            text = page["text"]
            page_num = page["page_number"]
            
            if not text:
                continue

            # Split the page text recursively
            raw_splits = self._split_text(text, self.separators)
            
            # Merge splits to handle overlap
            current_chunk = ""
            for split in raw_splits:
                if not current_chunk:
                    current_chunk = split
                elif len(current_chunk) + len(split) <= self.chunk_size:
                    current_chunk += " " + split
                else:
                    # Save current chunk
                    chunks.append({
                        "chunk_id": f"{doc_id}_c{chunk_id}",
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "text": current_chunk,
                        "chunk_index": chunk_id
                    })
                    chunk_id += 1
                    
                    # Prepare next chunk with overlap
                    # We take the overlap from the end of the current chunk
                    overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                    overlap_text = current_chunk[overlap_start:]
                    
                    # Ensure overlap starts at a word boundary if possible
                    space_idx = overlap_text.find(" ")
                    if space_idx != -1:
                        overlap_text = overlap_text[space_idx + 1:]

                    current_chunk = overlap_text + " " + split if overlap_text else split

            if current_chunk:
                chunks.append({
                    "chunk_id": f"{doc_id}_c{chunk_id}",
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "text": current_chunk,
                    "chunk_index": chunk_id
                })
                chunk_id += 1

        return chunks
