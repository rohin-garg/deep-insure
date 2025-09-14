import { InsuranceSection } from "@/utils/mockData";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

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

      const response = await fetch(`${API_BASE_URL}/get_full_summary?${params}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          // ngrok might require this header
          'ngrok-skip-browser-warning': 'true'
        }
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status} ${response.statusText}`);
      }

      const data: BackendSummaryResponse = await response.json();
      return transformSummaryResponse(data);
    } catch (error) {
      console.error('Error fetching summary:', error);
      throw error;
    }
  },

  // Placeholder for future endpoints
  async generateChatId(insurancePlanUrl: string): Promise<string> {
    const params = new URLSearchParams({
      insurance_plan_url: insurancePlanUrl
    });

    const response = await fetch(`${API_BASE_URL}/generate_chat_id?${params}`, {
      method: 'GET',
      headers: {
        'ngrok-skip-browser-warning': 'true'
      }
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data.id;
  },

  async getChatHistory(id: string): Promise<string[]> {
    const params = new URLSearchParams({ id });

    const response = await fetch(`${API_BASE_URL}/get_chat_history?${params}`, {
      method: 'GET',
      headers: {
        'ngrok-skip-browser-warning': 'true'
      }
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  },

  async askQuery(id: string, query: string): Promise<string> {
    const params = new URLSearchParams({ id, query });

    const response = await fetch(`${API_BASE_URL}/ask_query?${params}`, {
      method: 'GET',
      headers: {
        'ngrok-skip-browser-warning': 'true'
      }
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return response.text();
  }
};