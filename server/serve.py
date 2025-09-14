from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import uuid
import json
import requests
from typing import List, Dict, Any, Optional
import asyncio
import aiohttp
from rapidfuzz import fuzz
import anthropic
import os
from contextlib import contextmanager
from dotenv import load_dotenv
import hashlib
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pdfplumber
import io
import time
import sys

load_dotenv()

app = FastAPI(title="Insurance Plan Analysis API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_PATH = "insurance_analysis.db"
MAX_TOKENS = 2048

# Pydantic models
class ChatHistoryItem(BaseModel):
    prompt: str
    response: str

class SummaryPage(BaseModel):
    header: str
    text: str

class SummaryResponse(BaseModel):
    body: List[Dict[str, str]]

class ContextItem(BaseModel):
    url: str
    relevant_text: str

# Initialize database
def init_database():
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        
        # Create chat_sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                insurance_plan_url TEXT NOT NULL,
                chat_history TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create url_cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS url_cache (
                url TEXT PRIMARY KEY,
                context TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create insurance_context_cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insurance_context_cache (
                cache_key TEXT PRIMARY KEY,
                context_items TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create full_summary_cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS full_summary_cache (
                insurance_plan_url TEXT PRIMARY KEY,
                summary_data TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

# Initialize database on startup
init_database()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")
MODEL_ID = "claude-sonnet-4-20250514"

# Initialize Anthropic client
anthropic_client = anthropic.Anthropic()

# Helper functions
async def get_general_context(url: str) -> str:
    """Get raw text from a URL using Exa API"""
    try:
        # Check cache first
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT context FROM url_cache WHERE url = ?", (url,))
            cached = cursor.fetchone()
            if cached:
                return cached[0]
        
        # Make request to Exa API
        headers = {
            'x-api-key': EXA_API_KEY,
            'Content-Type': 'application/json'
        }
        
        payload = {
            "urls": [url],
            "text": True,
            "context": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post('https://api.exa.ai/contents', 
                                  headers=headers, 
                                  json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('results') and len(data['results']) > 0:
                        context = data['results'][0].get('text', '')
                        
                        # Cache the result
                        with get_db_connection() as conn:
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT OR REPLACE INTO url_cache (url, context) VALUES (?, ?)",
                                (url, context)
                            )
                            conn.commit()
                        
                        return context
                    
        return ""
    except Exception as e:
        print(f"Error fetching context for {url}: {e}")
        return ""

def generate_personalized_search_keywords(context_items: List[ContextItem]) -> List[str]:
    """Generate personalized search keywords based on the specific insurance plan context"""
    try:
        # Combine all context text
        combined_context = "\n\n".join([item.relevant_text for item in context_items])
        
        # Limit context to avoid token limits
        if len(combined_context) > 8000:
            combined_context = combined_context[:8000]
        
        prompt = f"""Based on this specific insurance plan information, generate 6 targeted search queries that would help find real user experiences, reviews, and discussions about this particular plan or very similar plans.

Insurance Plan Context:
{combined_context}

Generate search queries that are:
1. Specific to this plan type, provider, or similar coverage
2. Focused on real user experiences and reviews
3. Targeted at finding discussions on Reddit, forums, or review sites
4. Include specific plan features, benefits, or concerns mentioned in the context

Return exactly 6 search queries, one per line, without numbering or bullets. Focus on the specific plan name, provider, coverage details, and unique features mentioned in the context."""

        response = anthropic_client.messages.create(
            model=MODEL_ID,
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse the response into individual queries
        queries = [q.strip() for q in response.content[0].text.strip().split('\n') if q.strip()]
        
        # Ensure we have exactly 6 queries, pad with generic ones if needed
        while len(queries) < 6:
            queries.append("insurance plan review user experience")
        
        return queries[:6]
        
    except Exception as e:
        print(f"Error generating personalized keywords: {e}")
        # Fallback to generic queries
        return [
            "reddit insurance plan review user experience",
            "forum discussion insurance coverage real experience", 
            "customer review insurance plan pros cons",
            "reddit healthcare insurance worth it",
            "user experience insurance claim process",
            "forum insurance plan comparison real users"
        ]

async def get_diverse_sources_context(insurance_plan_url: str, base_query: str, context_items: List[ContextItem] = None) -> List[ContextItem]:
    """Get additional context from diverse sources like Reddit, forums, and user experiences using Exa API"""
    try:
        # Extract insurance plan name/type from URL for better search queries
        plan_identifier = insurance_plan_url.split('/')[-1] if '/' in insurance_plan_url else insurance_plan_url
        
        # Generate personalized search queries if context is provided
        if context_items:
            print("Generating personalized search keywords based on insurance plan context...")
            search_queries = generate_personalized_search_keywords(context_items)
            print(f"Generated personalized queries: {search_queries}")
        else:
            # Fallback to generic search queries
            search_queries = [
                f"reddit insurance plan review {base_query} user experience",
                f"forum discussion {base_query} insurance coverage real experience",
                f"customer review {base_query} insurance plan pros cons",
                f"reddit healthcare insurance {base_query} worth it",
                f"user experience {base_query} insurance claim process",
                f"forum {base_query} insurance plan comparison real users"
            ]
        
        # Search using Exa API for each query
        headers = {
            'x-api-key': EXA_API_KEY,
            'Content-Type': 'application/json'
        }
        
        diverse_context_items = []
        
        async with aiohttp.ClientSession() as session:
            for query in search_queries:
                try:
                    # Use Exa search API to find relevant URLs
                    search_payload = {
                        "query": query,
                        "numResults": 3,
                        "includeDomains": ["reddit.com", "quora.com", "healthline.com", "consumerreports.org"],
                        "type": "neural"
                    }
                    
                    async with session.post('https://api.exa.ai/search', 
                                          headers=headers, 
                                          json=search_payload) as search_response:
                        if search_response.status == 200:
                            search_data = await search_response.json()
                            urls = [result['url'] for result in search_data.get('results', [])]
                            
                            if urls:
                                # Get content from found URLs
                                content_payload = {
                                    "urls": urls,
                                    "text": True,
                                    "highlights": {
                                        "numSentences": 3,
                                        "highlightsPerUrl": 3,
                                        "query": base_query
                                    }
                                }
                                
                                async with session.post('https://api.exa.ai/contents',
                                                      headers=headers,
                                                      json=content_payload) as content_response:
                                    if content_response.status == 200:
                                        content_data = await content_response.json()
                                        
                                        for result in content_data.get('results', []):
                                            if result.get('text'):
                                                # Use highlights if available, otherwise use raw text excerpt
                                                relevant_text = ""
                                                if result.get('highlights'):
                                                    relevant_text = " ".join(result['highlights'])
                                                else:
                                                    # Take first 1000 chars of text
                                                    relevant_text = result['text'][:1000]
                                                
                                                if relevant_text.strip():
                                                    diverse_context_items.append(ContextItem(
                                                        url=result['url'],
                                                        relevant_text=relevant_text
                                                    ))
                    
                    # Add small delay between requests to be respectful
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error searching for query '{query}': {e}")
                    continue
        
        print(diverse_context_items)
        return diverse_context_items
        
    except Exception as e:
        print(f"Error getting diverse sources context: {e}")
        return []

def extract_keywords_with_llm(query: str) -> List[str]:
    """Extract keywords from query using LLM"""
    try:
        response = anthropic_client.messages.create(
            model=MODEL_ID,
            max_tokens=100,
            messages=[
                {"role": "user", "content": f"Extract 5-10 key search terms from this query. Return only the terms separated by commas: {query}"}
            ]
        )
        
        keywords = response.content[0].text.strip().split(',')
        return [kw.strip().lower() for kw in keywords]
    except:
        # Fallback: simple keyword extraction
        return query.lower().split()

def raw_search(query: str, raw_text: str) -> List[str]:
    """Perform fuzzy search on raw text and return top 10 blocks"""
    if not raw_text:
        return []
    
    keywords = extract_keywords_with_llm(query)
    
    # Split text into blocks of ~500 characters
    block_size = 500
    blocks = []
    for i in range(0, len(raw_text), block_size):
        block = raw_text[i:i + block_size]
        blocks.append(block)
    
    # Score each block
    scored_blocks = []
    for block in blocks:
        score = 0
        block_lower = block.lower()
        
        for keyword in keywords:
            # Use partial ratio for fuzzy matching
            fuzzy_score = fuzz.partial_ratio(keyword, block_lower)
            score += fuzzy_score
            
            # Jaccard similarity
            block_words = set(block_lower.split())
            keyword_words = set(keyword.split())
            if keyword_words and block_words:
                jaccard = len(keyword_words.intersection(block_words)) / len(keyword_words.union(block_words))
                score += jaccard * 100
        
        scored_blocks.append((score, block))
    
    # Sort by score and return top 10
    scored_blocks.sort(key=lambda x: x[0], reverse=True)
    return [block for score, block in scored_blocks[:10]]

async def scrape_links_with_browser(url: str) -> List[str]:
    """Scrape links from a page using headless browser, filtering for UHC Medicare alphadog PDFs"""
    try:
        # Setup Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Navigate to the page
            driver.get(url)
            
            # Wait for the page to load and links to appear (up to 10 seconds)
            time.sleep(5)  # Give time for dynamic content to load
            
            # Find all links on the page
            link_elements = driver.find_elements(By.TAG_NAME, "a")
            
            # Extract href attributes and filter for UHC Medicare alphadog links
            filtered_links = []
            for element in link_elements:
                href = element.get_attribute("href")
                if href and href.startswith("https://www.uhc.com/medicare/alphadog"):
                    filtered_links.append(href)
            
            # Remove duplicates
            filtered_links = list(set(filtered_links))
            
            return filtered_links
            
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"Error scraping links from {url}: {e}")
        return []

def is_english_text(text: str) -> bool:
    """Check if text is likely in English using simple heuristics"""
    if not text or len(text.strip()) < 50:  # Need minimum text to analyze
        return False
    
    # Convert to lowercase for analysis
    text_lower = text.lower()
    words = text_lower.split()
    
    if len(words) < 10:  # Need minimum words to analyze
        return False
    
    # Common English words to check for
    english_indicators = ['the', 'and', 'or', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'with', 'as', 'on', 'by']
    
    # Count occurrences of English indicators
    indicator_count = 0
    for indicator in english_indicators:
        indicator_count += text_lower.count(f' {indicator} ') + text_lower.count(f'{indicator} ')
    
    # Calculate percentage of English indicators
    english_percentage = indicator_count / len(words)
    
    # Also check for basic English letter patterns
    total_chars = len([c for c in text_lower if c.isalpha()])
    if total_chars == 0:
        return False
    
    # Count common English letters
    common_english_letters = 'etaoinshrdlu'
    common_letter_count = sum(text_lower.count(letter) for letter in common_english_letters)
    letter_percentage = common_letter_count / total_chars
    
    # Consider it English if either metric suggests it
    return english_percentage > 0.05 or letter_percentage > 0.4

async def extract_pdf_text(pdf_url: str) -> str:
    """Extract text from a PDF URL using browser session to handle authentication"""
    try:
        # Setup Chrome options for headless browsing
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Navigate to the PDF URL using the browser (this handles cookies/session)
            driver.get(pdf_url)
            
            # Wait a moment for the PDF to load
            time.sleep(3)
            
            # Get the page source (this might contain the PDF content or redirect info)
            page_source = driver.page_source
            
            # If we can get the current URL after potential redirects
            current_url = driver.current_url
            
            # Try to download using requests with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/pdf,application/octet-stream,*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Get cookies from the browser session
            cookies = {}
            for cookie in driver.get_cookies():
                cookies[cookie['name']] = cookie['value']
            
            # Try downloading with aiohttp using browser headers and cookies
            async with aiohttp.ClientSession(headers=headers, cookies=cookies) as session:
                async with session.get(current_url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'pdf' in content_type or pdf_url.endswith('.pdf'):
                            pdf_content = await response.read()
                            
                            # First, check the first 5 pages for English content
                            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                                # Extract text from first 5 pages to check language
                                sample_text = ""
                                pages_to_check = min(5, len(pdf.pages))
                                
                                for i in range(pages_to_check):
                                    page_text = pdf.pages[i].extract_text()
                                    if page_text:
                                        sample_text += page_text + "\n"
                                
                                # Check if the sample text is in English
                                if not is_english_text(sample_text):
                                    print(f"PDF {pdf_url} does not appear to be in English, skipping")
                                    return ""
                                
                                # If English, extract text from all pages
                                text_content = ""
                                for page in pdf.pages:
                                    page_text = page.extract_text()
                                    if page_text:
                                        text_content += page_text + "\n"
                                
                                return text_content.strip()
                        else:
                            print(f"URL {pdf_url} does not return PDF content (content-type: {content_type})")
                            return ""
                    else:
                        print(f"Failed to download PDF from {pdf_url}: HTTP {response.status}")
                        return ""
            
        finally:
            driver.quit()
            
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_url}: {e}")
        return ""

async def get_insurance_context(insurance_plan_url: str, query: str) -> List[ContextItem]:
    """Get relevant context for insurance plan URL with caching"""
    try:
        # Create cache key from URL and query
        cache_key = hashlib.md5(f"{insurance_plan_url}:{query}".encode()).hexdigest()
        
        # Check cache first
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT context_items FROM insurance_context_cache WHERE cache_key = ?", (cache_key,))
            cached = cursor.fetchone()
            if cached:
                # Deserialize cached context items
                cached_data = json.loads(cached[0])
                return [ContextItem(**item) for item in cached_data]
        
        # Get general context from the URL
        raw_text = await get_general_context(insurance_plan_url)
        
        context_items = []
        
        # Add the full raw text from get_general_context (already filtered, don't use raw_search on it)
        if raw_text:
            context_items.append(ContextItem(
                url=insurance_plan_url,
                relevant_text=raw_text.strip()
            ))
        
        # Scrape PDF links using headless browser
        pdf_links = await scrape_links_with_browser(insurance_plan_url)
        
        # Process each PDF link
        for pdf_url in pdf_links:
            try:
                # Extract text from PDF
                print('extracting pdf', pdf_url)
                pdf_text = await extract_pdf_text(pdf_url)
                if pdf_text:
                    # print(pdf_url, pdf_text)
                    # Use raw_search on the PDF text to find relevant blocks
                    relevant_blocks = raw_search(query, pdf_text)
                    
                    # Add each relevant block as a context item with the PDF URL
                    for block in relevant_blocks:
                        context_items.append(ContextItem(
                            url=pdf_url,
                            relevant_text=block.strip()
                        ))
                        
            except Exception as e:
                print(f"Error processing PDF {pdf_url}: {e}")
                continue
        
        # Cache the results
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Serialize context items for storage
            serialized_items = [item.dict() for item in context_items]
            cursor.execute(
                "INSERT OR REPLACE INTO insurance_context_cache (cache_key, context_items) VALUES (?, ?)",
                (cache_key, json.dumps(serialized_items))
            )
            conn.commit()
        
        return context_items
    except Exception as e:
        print(f"Error getting insurance context: {e}")
        return []

def generate_summary_with_llm(context_items: List[ContextItem]) -> Dict[str, Any]:
    """Generate summary using LLM and parse markdown headers"""
    if not context_items:
        return {"body": [{"page_1_header": "Summary", "page_1_text": "No context available for summary generation."}]}
    
    try:
        # Combine all context
        combined_text = "\n\n".join([item.relevant_text for item in context_items])
        
        # Create cache key for the raw markdown output
        markdown_cache_key = hashlib.md5(combined_text.encode()).hexdigest()
        
        # Check if we have cached markdown output
        conn = sqlite3.connect('insurance_analysis.db')
        cursor = conn.cursor()
        
        # Create markdown cache table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markdown_output_cache (
                cache_key TEXT PRIMARY KEY,
                markdown_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Try to get cached markdown
        cursor.execute("SELECT markdown_text FROM markdown_output_cache WHERE cache_key = ?", (markdown_cache_key,))
        cached_result = cursor.fetchone()
        
        if cached_result:
            print("Using cached markdown output")
            markdown_text = cached_result[0]
        else:
            print("Generating new markdown with Claude")
            prompt = f"""
            Please analyze the following insurance plan information and create a structured summary in markdown format.
            Provide detailed content under each header. You should include ALL relevant information, be as comprehensive and detailed as possible, try to make the summary descriptive and readable even if the consumer is unaware of what the different insurance terms mean (use your knowledge to extrapolate). I want long, iimpressive, information-filled sections.
            
            Insurance Plan Information:
            {combined_text}
            
            Create a comprehensive summary covering key aspects like coverage, costs, benefits, etc.
            Return only markdown with no additional text. Use single hashtags for the main sections (there should be a lot of these, one for each of the important sections of the policy, not just the title). I'm expecting 5-10 sections with one hashtag, and length descriptions for each section.
            Each one of these sections should also have subheadings. I want a long, nicely structured document (tables would also be great)
            All text should be under at least one header.
            """
            
            response = anthropic_client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            markdown_text = response.content[0].text.strip()
            
            # Cache the markdown output
            cursor.execute(
                "INSERT OR REPLACE INTO markdown_output_cache (cache_key, markdown_text) VALUES (?, ?)",
                (markdown_cache_key, markdown_text)
            )
            conn.commit()
        
        conn.close()
        
        print("===== BEGIN MARKDOWN TEXT =====")
        print(markdown_text)
        print("===== END MARKDOWN TEXT =====")
        
        # Parse markdown to extract sections with single # headers
        sections = []
        current_header = None
        current_content = []
        
        for line in markdown_text.split('\n'):
            # Check if line is a single # header (not ## or ###)
            if line.startswith("# ") and not line.startswith("## "):
                continue
            elif line.startswith('## ') and not line.startswith('### '):
                # Save previous section if exists
                if current_header is not None and len('\n'.join(current_content).strip()) > 0:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_content).strip()
                    })
                
                # Start new section
                current_header = line[2:].strip()  # Remove '# ' prefix
                current_content = []
            else:
                # Add line to current content
                if current_header is not None:
                    current_content.append(line)
        
        # Don't forget the last section
        if current_header is not None:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_content).strip()
            })
        
        # Convert to required JSON format
        body = []
        for i, section in enumerate(sections, 1):
            body.append({
                f"page_{i}_header": section['header'],
                f"page_{i}_text": section['content']
            })
        
        # If no sections found, create a default one
        if not body:
            body = [{"page_1_header": "Summary", "page_1_text": markdown_text}]
        
        return {"body": body}
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return {"body": [{"page_1_header": "Summary", "page_1_text": f"Error generating summary: {str(e)}"}]}

def generate_answer_with_llm(query: str, context_items: List[ContextItem]) -> str:
    """Generate answer to query using LLM with citations"""
    if not context_items:
        return "No relevant context found for your query."
    
    try:
        # Combine context with URLs for citations
        context_with_citations = []
        for i, item in enumerate(context_items):
            context_with_citations.append(f"[Source {i+1}]({item.url}): {item.relevant_text}")
        
        combined_context = "\n\n".join(context_with_citations)
        
        prompt = f"""
        Based on the following context about an insurance plan, answer the user's question. Use markdown to format the text.
        Include citations in markdown format using the source links provided. When citing a link, pick the most representative word/phrase present in the text and make that the place holder text of the link. For example, if the text is "The reason this plan is good is dental insurance" the markdown for the link should be ["dental insurance"](link).
        
        Context:
        {combined_context}
        
        Question: {query}
        
        Provide a comprehensive answer with relevant citations.
        """
        
        response = anthropic_client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# API Endpoints

@app.get("/generate_chat_id")
async def generate_chat_id(insurance_plan_url: str = Query(..., description="URL of the insurance plan")):
    """Generate a unique chat ID for an insurance plan URL"""
    try:
        chat_id = str(uuid.uuid4())
        
        # Initialize empty chat history
        initial_history = []
        
        # Store in database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_sessions (id, insurance_plan_url, chat_history) VALUES (?, ?, ?)",
                (chat_id, insurance_plan_url, json.dumps(initial_history))
            )
            conn.commit()
        
        return {"id": chat_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating chat ID: {str(e)}")

@app.get("/get_chat_history")
async def get_chat_history(id: str = Query(..., description="Chat session ID")):
    """Get chat history for a given chat ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chat_history FROM chat_sessions WHERE id = ?", (id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Chat session not found")
            
            chat_history = json.loads(result[0])
            
            # Format as list of strings (alternating prompts and responses)
            formatted_history = []
            for item in chat_history:
                formatted_history.append(f"**User:** {item['prompt']}")
                formatted_history.append(f"**Assistant:** {item['response']}")
            
            return formatted_history
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.get("/get_full_summary")
async def get_full_summary(insurance_plan_url: str = Query(..., description="URL of the insurance plan")):
    """Get full summary of an insurance plan"""
    try:
        # Check cache first
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT summary_data FROM full_summary_cache WHERE insurance_plan_url = ?", (insurance_plan_url,))
            cached_result = cursor.fetchone()
            
            if cached_result:
                print(f"Returning cached summary for {insurance_plan_url}")
                return json.loads(cached_result[0])
        
        # Get context for the insurance plan (using a general query)
        base_query = "insurance plan summary coverage benefits costs"
        context_items = await get_insurance_context(insurance_plan_url, base_query)
        
        if not context_items:
            return {"body": [{"page_1_header": "No Data", "page_1_text": "Unable to retrieve information for this insurance plan URL."}]}
        
        # Get additional context from diverse sources (Reddit, forums, user experiences)
        print("Gathering additional context from diverse sources...")
        diverse_context = await get_diverse_sources_context(insurance_plan_url, base_query, context_items)
        
        # Combine original context with diverse sources
        all_context_items = context_items + diverse_context
        
        # Generate summary using LLM with enhanced context
        summary = generate_summary_with_llm(all_context_items)
        
        # Cache the result
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO full_summary_cache (insurance_plan_url, summary_data) VALUES (?, ?)",
                (insurance_plan_url, json.dumps(summary))
            )
            conn.commit()
            print(f"Cached summary for {insurance_plan_url}")
        
        return summary
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.get("/ask_query")
async def ask_query(id: str = Query(..., description="Chat session ID"), 
                   query: str = Query(..., description="User query")):
    """Answer a query about the insurance plan"""
    try:
        # Get chat session info
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT insurance_plan_url, chat_history FROM chat_sessions WHERE id = ?", (id,))
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Chat session not found")
            
            insurance_plan_url, chat_history_json = result
            chat_history = json.loads(chat_history_json)
        
        # Get relevant context for the query
        context_items = await get_insurance_context(insurance_plan_url, query)
        
        # Generate answer using LLM
        answer = generate_answer_with_llm(query, context_items)
        
        # Update chat history
        chat_history.append({
            "prompt": query,
            "response": answer
        })
        
        # Save updated chat history
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE chat_sessions SET chat_history = ? WHERE id = ?",
                (json.dumps(chat_history), id)
            )
            conn.commit()
        
        return {"answer": answer}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Additional utility endpoints

@app.get("/raw_search")
async def raw_search_endpoint(query: str = Query(..., description="Search query"),
                             raw_text: str = Query(..., description="Raw text to search")):
    """Perform raw search on provided text"""
    try:
        results = raw_search(query, raw_text)
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insurance Plan Analysis API",
        "version": "1.0.0",
        "endpoints": [
            "/generate_chat_id",
            "/get_chat_history",
            "/get_full_summary", 
            "/ask_query",
            "/raw_search"
        ]
    }

def test_functions():
    URL = "https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049"
    query = "what benefits does this plan offer? what important things are NOT covered?"
    # res = asyncio.run(get_general_context(URL))
    # res = asyncio.run(get_insurance_context(URL, query))
    res = asyncio.run(get_full_summary(URL))
    # obj = asyncio.run(generate_chat_id(URL))
    # id = obj["id"]
    # res = asyncio.run(ask_query(id, query))
    print(res)
    import code; code.interact(local=dict(globals(), **locals()))

if __name__ == "__main__":
    # test_functions()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)