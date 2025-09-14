Endpoints
- `generate_chat_id(insurance_plan_url: string)`: GET -> returns unique id: string
- `get_chat_history(id: string)`: GET -> returns history of the chat (markdown w/ citations, citation includes relevant words): list[string]
    - citation is in standard markdown format: `[key words](link)`
- `get_full_summary(insurance_plan_url: string)`: GET -> returns json of raw summary of the insurance plan (markdown w/ regular citations)
    - json schema: 
    ```
    {"body":[
        {'page_1_header': string, 'page_1_text': string},
        {'page_2_header': string, 'page_2_text': string},
        ...
    ]}
    ```
- `ask_query(id: string, query: string)`: GET -> returns answer to the question (markdown w/ citations, citation includes relevant words): string
    

General server architecture
- have a sql table to store the ids: list of chat history
    - cols: `id: string, insurance_plan_url: string, chat_history: list[prompt, response]`
- sql table to store the cached context from a URL (both `get_general_context` and `get_insurance_context` data will be cached)
    - cols: url: string, context: string
- `get_general_context()`: just queries exa's api to get raw text from the page
- `raw_search(query: string, raw_text: string)`: GET -> returns raw search results using a much cheaper solution that pasting all the queries in
    - uses the query/LLM to return a few keywords/phrases, for which it does a raw search
        - raw search uses rapidfuzz for fuzz.partial_ratio, jaccard_similarity to filter out the top 10 blocks (each block is ~500 chars)
- `get_insurance_context(insurance_plan_url, query)`: special one that returns the context for insurance_plan_url -> list[{url, relevant_text}]
    - calls `get_general_context()`
    - we also collate the text from all the raw links that are returned from exa's API and do some custom search algorithm
        - use an LLM to generate a list of key words, do a raw search on the text, and combine the text into context (taking the header of the page + the surrounding block)
- `get_full_summary(insurance_plan_url)`: just puts the context from get_insurance_context() into chatgpt, and returns a markdown file
    - prompt: ``



this is an example of a request to exa's API
```
curl -X POST 'https://api.exa.ai/contents' \
  -H 'x-api-key: d6c6c02e-854f-4d3a-ba67-0121b3c796fd' \
  -H 'Content-Type: application/json' \
  -d '{
    "urls": ["https://www.uhc.com/medicare/health-plans/details.html/02139/017/H8768055002/2025?WT.mc_id=8031049"],
    "text": true,
    "context": true,
    "highlights": {
        "numSentences": 2,
        "highlightsPerUrl": 5,
        "query": "",
    },
    "summary": {
        "query": "",
        "schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "Property 1": {
                    "type": "string",
                    "description": "Description"
                },
                "Property 2": {
                    "type": "string",
                    "enum": ["option 1", "option 2", "option 3"],
                    "description": "Description"
                }
            },
            "required": ["Property 1"]
        }
    }
  }'
```
^ request outputs @request_output.json