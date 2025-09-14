import { InsuranceSection } from "@/utils/mockData";

// Remove trailing slash if present
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '');

// Backend response types
interface BackendSummaryPage {
  [key: string]: string; // e.g., "page_1_header": "string", "page_1_text": "string"
}

interface BackendSummaryResponse {
  body: BackendSummaryPage[];
}

// Transform backend response to frontend format
function transformSummaryResponse(response: BackendSummaryResponse): InsuranceSection[] {
  const sections: InsuranceSection[] = [];

  response.body.forEach((page) => {
    // Extract page number and create section
    const pageEntries = Object.entries(page);
    const headerEntry = pageEntries.find(([key]) => key.includes('_header'));
    const textEntry = pageEntries.find(([key]) => key.includes('_text'));

    if (headerEntry && textEntry) {
      const pageNumber = headerEntry[0].match(/page_(\d+)_header/)?.[1] || '1';
      const header = headerEntry[1];
      const text = textEntry[1];

      // Create ID from header (kebab-case)
      const id = header.toLowerCase()
        .replace(/[^a-z0-9\s]/g, '')
        .replace(/\s+/g, '-');

      sections.push({
        id: id || `section-${pageNumber}`,
        header,
        text
      });
    }
  });

  return sections;
}

// API functions
export const api = {
  async getFullSummary(insurancePlanUrl: string): Promise<InsuranceSection[]> {
    try {
      const params = new URLSearchParams({
        insurance_plan_url: insurancePlanUrl
      });

      const url = `${API_BASE_URL}/get_full_summary?${params}`;
      console.log('Fetching summary from:', url);
      console.log('With params:', { insurance_plan_url: insurancePlanUrl });

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // ngrok might require this header
          'ngrok-skip-browser-warning': 'true'
        }
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error response body:', errorText);
        throw new Error(`API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data: BackendSummaryResponse = await response.json();
      console.log('Received data:', data);

      const transformed = transformSummaryResponse(data);
      console.log('Transformed sections:', transformed);

      return transformed;
    } catch (error) {
      console.error('Error fetching summary:', error);
      throw error;
    }
  },

  async generateChatId(insurancePlanUrl: string): Promise<string> {
    try {
      const params = new URLSearchParams({
        insurance_plan_url: insurancePlanUrl
      });

      console.log('Generating chat ID for:', insurancePlanUrl);

      const response = await fetch(`${API_BASE_URL}/generate_chat_id?${params}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error generating chat ID:', errorText);
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      console.log('Generated chat ID:', data.id);
      return data.id;
    } catch (error) {
      console.error('Error in generateChatId:', error);
      throw error;
    }
  },

  async getChatHistory(id: string): Promise<string[]> {
    try {
      const params = new URLSearchParams({ id });

      console.log('Fetching chat history for ID:', id);

      const response = await fetch(`${API_BASE_URL}/get_chat_history?${params}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        }
      });

      if (!response.ok) {
        if (response.status === 404) {
          console.log('Chat session not found, returning empty history');
          return [];
        }
        throw new Error(`API error: ${response.status}`);
      }

      const history = await response.json();
      console.log('Retrieved chat history:', history);
      return history;
    } catch (error) {
      console.error('Error in getChatHistory:', error);
      throw error;
    }
  },

  async askQuery(id: string, query: string): Promise<string> {
    try {
      const params = new URLSearchParams({ id, query });

      console.log('Asking query:', query, 'for chat ID:', id);

      const response = await fetch(`${API_BASE_URL}/ask_query?${params}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true'
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Error asking query:', errorText);
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      console.log('Query response:', data);
      return data.answer || data;
    } catch (error) {
      console.error('Error in askQuery:', error);
      throw error;
    }
  }
};