import { useEffect, useState, useRef, useCallback } from "react";
import { useSearchParams, Link } from "react-router-dom";
import ReactMarkdown from 'react-markdown';
import { Header } from "@/components/Header";
import { ArrowLeft, User, Bot, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatBar } from "@/components/ChatBar";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";
import { mockAIResponse, mockFollowUpResponses, mockSourceCards, mockChatHistory, generateMockChatId } from "@/utils/mockData";

interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  isLoading?: boolean;
  isTyping?: boolean;
  displayContent?: string;
}

interface SourceCard {
  title: string;
  url: string;
  snippet: string;
  type: string;
}

const Chat = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sources, setSources] = useState<SourceCard[]>([]);
  const [isLoadingNewMessage, setIsLoadingNewMessage] = useState(false);
  const [chatId, setChatId] = useState<string | null>(null);
  const [insuranceUrl, setInsuranceUrl] = useState<string>('');
  const [isLoadingSources, setIsLoadingSources] = useState(false);
  const [visibleSources, setVisibleSources] = useState<Set<number>>(new Set());
  const [dividerPosition, setDividerPosition] = useState(45); // Percentage for left panel
  const [isDragging, setIsDragging] = useState(false);
  const [collapsedSources, setCollapsedSources] = useState<Set<number>>(new Set());
  const [highlightedSource, setHighlightedSource] = useState<number | null>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const sourcesScrollRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const { toast } = useToast();

  const handleInitialQuestion = useCallback(async (question: string, url: string, existingChatId: string | null) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: question
    };
    const aiMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'ai',
      content: '',
      isLoading: true
    };

    setMessages([userMessage, aiMessage]);
    setIsLoadingNewMessage(true);

    try {
      // Generate chat ID if we don't have one
      let currentChatId = existingChatId;
      if (!currentChatId) {
        currentChatId = await api.generateChatId(url);
        setChatId(currentChatId);
        // Update URL with chat ID
        setSearchParams({ q: question, url, chat: currentChatId });
      }

      // Ask the question
      console.log('About to call askQuery...');
      const response = await api.askQuery(currentChatId, question);
      console.log('Got response from askQuery:', response);

      // Update the AI message with response
      setMessages(prev => prev.map(msg =>
        msg.id === aiMessage.id
          ? { ...msg, content: response, isLoading: false, displayContent: '' }
          : msg
      ));

      // Parse sources from response
      const extractedSources = extractSourcesFromResponse(response);
      setSources(extractedSources);

      // Start typing animation
      typeMessage(aiMessage.id, response);
    } catch (error) {
      console.error('Error processing initial question:', error);
      console.warn('🔄 API unavailable - falling back to mock data for development');

      // Fallback to mock data
      setTimeout(() => {
        // Generate mock chat ID if we don't have one
        let currentChatId = existingChatId;
        if (!currentChatId) {
          currentChatId = generateMockChatId();
          setChatId(currentChatId);
          setSearchParams({ q: question, url, chat: currentChatId });
        }

        // Use mock response
        setMessages(prev => prev.map(msg =>
          msg.id === aiMessage.id
            ? { ...msg, content: mockAIResponse, isLoading: false, displayContent: '' }
            : msg
        ));

        // Use mock sources
        animateSourcesLoading(mockSourceCards);

        // Start typing animation with mock response
        typeMessage(aiMessage.id, mockAIResponse);
      }, 1500); // Simulate API delay
    } finally {
      setIsLoadingNewMessage(false);
    }
  }, [setSearchParams, toast]);

  // Initialize chat from URL params
  const hasInitializedRef = useRef(false);

  useEffect(() => {
    const initializeChat = async () => {
      if (hasInitializedRef.current) return;
      hasInitializedRef.current = true;

      const q = searchParams.get("q");
      const url = searchParams.get("url");
      const existingChatId = searchParams.get("chat");

      console.log('Initializing chat with:', { q, url, existingChatId });

      if (url) {
        setInsuranceUrl(url);
      }

      // If we have an existing chat ID, load its history
      if (existingChatId) {
        setChatId(existingChatId);
        try {
          const history = await api.getChatHistory(existingChatId);
          console.log('Loaded chat history:', history);
          // Parse history into messages
          const parsedMessages: ChatMessage[] = [];
          for (let i = 0; i < history.length; i++) {
            const msg = history[i];
            if (msg.startsWith('**User:**')) {
              parsedMessages.push({
                id: `history-${i}`,
                type: 'user',
                content: msg.replace('**User:** ', '')
              });
            } else if (msg.startsWith('**Assistant:**')) {
              parsedMessages.push({
                id: `history-${i}`,
                type: 'ai',
                content: msg.replace('**Assistant:** ', '')
              });
            }
          }
          setMessages(parsedMessages);
        } catch (error) {
          console.error('Error loading chat history:', error);
          console.warn('🔄 API unavailable - using mock chat history for development');

          // Fallback to mock chat history
          const parsedMessages: ChatMessage[] = [];
          for (let i = 0; i < mockChatHistory.length; i++) {
            const msg = mockChatHistory[i];
            if (msg.startsWith('**User:**')) {
              parsedMessages.push({
                id: `history-${i}`,
                type: 'user',
                content: msg.replace('**User:** ', '')
              });
            } else if (msg.startsWith('**Assistant:**')) {
              parsedMessages.push({
                id: `history-${i}`,
                type: 'ai',
                content: msg.replace('**Assistant:** ', '')
              });
            }
          }
          setMessages(parsedMessages);
        }
      }

      // If we have a new question and insurance URL
      if (q && url) {
        console.log('Processing initial question');

        const userMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'user',
          content: q
        };
        const aiMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'ai',
          content: '',
          isLoading: true
        };

        setMessages([userMessage, aiMessage]);
        setIsLoadingNewMessage(true);

        try {
          // Generate chat ID if we don't have one
          let currentChatId = existingChatId;
          if (!currentChatId) {
            currentChatId = await api.generateChatId(url);
            setChatId(currentChatId);
            // Update URL with chat ID
            setSearchParams({ q, url, chat: currentChatId });
          }

          // Ask the question
          console.log('About to call askQuery...');
          const response = await api.askQuery(currentChatId, q);
          console.log('Got response from askQuery:', response);

          // Update the AI message with response
          setMessages(prev => prev.map(msg =>
            msg.id === aiMessage.id
              ? { ...msg, content: response, isLoading: false, displayContent: '' }
              : msg
          ));

          // Parse sources from response
          console.log('Extracting sources from response:', response.substring(0, 500) + '...');
          const extractedSources = extractSourcesFromResponse(response);
          console.log('Extracted sources:', extractedSources);

          if (extractedSources.length > 0) {
            animateSourcesLoading(extractedSources);
          } else {
            setSources(extractedSources);
          }

          // Start typing animation
          typeMessage(aiMessage.id, response);
        } catch (error) {
          console.error('Error processing initial question:', error);
          console.warn('🔄 API unavailable - falling back to mock data for development');

          // Fallback to mock data
          setTimeout(() => {
            // Generate mock chat ID if we don't have one
            let currentChatId = existingChatId;
            if (!currentChatId) {
              currentChatId = generateMockChatId();
              setChatId(currentChatId);
              setSearchParams({ q, url, chat: currentChatId });
            }

            // Use mock response
            setMessages(prev => prev.map(msg =>
              msg.id === aiMessage.id
                ? { ...msg, content: mockAIResponse, isLoading: false, displayContent: '' }
                : msg
            ));

            // Use mock sources
            animateSourcesLoading(mockSourceCards);

            // Start typing animation with mock response
            typeMessage(aiMessage.id, mockAIResponse);
          }, 1500); // Simulate API delay
        } finally {
          setIsLoadingNewMessage(false);
        }
      }
    };

    initializeChat();
  }, [searchParams]);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging || !containerRef.current) return;
    
    const containerRect = containerRef.current.getBoundingClientRect();
    const newPosition = ((e.clientX - containerRect.left) / containerRect.width) * 100;
    
    // Constrain between 20% and 80% to prevent panels from becoming too small
    const constrainedPosition = Math.min(Math.max(newPosition, 20), 80);
    setDividerPosition(constrainedPosition);
  }, [isDragging]);

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, handleMouseMove]);

  // Cleanup typing timeout on unmount
  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, []);

  const toggleSourceCollapse = (index: number) => {
    setCollapsedSources(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  const animateSourcesLoading = (sourceCards: SourceCard[]) => {
    setIsLoadingSources(true);
    setSources(sourceCards);
    setVisibleSources(new Set()); // Start with all sources hidden
    setCollapsedSources(new Set(sourceCards.map((_, index) => index))); // Start all collapsed
    
    // Animate sources appearing one by one
    sourceCards.forEach((_, index) => {
      setTimeout(() => {
        setVisibleSources(prev => new Set([...prev, index]));
      }, index * 200); // 200ms delay between each source
    });
    
    // Finish loading after all sources are visible
    setTimeout(() => {
      setIsLoadingSources(false);
    }, sourceCards.length * 200 + 500);
  };

  const handleCitationClick = (sourceIndex: number) => {
    setHighlightedSource(sourceIndex);

    // Expand the source card if it's collapsed
    setCollapsedSources(prev => {
      const newSet = new Set(prev);
      newSet.delete(sourceIndex); // Remove from collapsed set to expand it
      return newSet;
    });

    // Scroll to the source element
    const sourceElement = document.querySelector(`[data-source-index="${sourceIndex}"]`);
    if (sourceElement && sourcesScrollRef.current) {
      sourceElement.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }

    // Auto-clear highlight after 3 seconds
    setTimeout(() => {
      setHighlightedSource(null);
    }, 3000);
  };

  const typeMessage = (messageId: string, fullContent: string) => {
    const words = fullContent.split(' ');
    let currentWordIndex = 0;
    const typingSpeed = 6; // milliseconds per word (4x faster than before)

    const typeNextWord = () => {
      if (currentWordIndex < words.length) {
        const displayContent = words.slice(0, currentWordIndex + 1).join(' ');

        setMessages(prev => prev.map(msg =>
          msg.id === messageId
            ? { ...msg, displayContent, isTyping: true }
            : msg
        ));

        currentWordIndex++;
        typingTimeoutRef.current = setTimeout(typeNextWord, typingSpeed);
      } else {
        // Typing complete
        setMessages(prev => prev.map(msg =>
          msg.id === messageId
            ? { ...msg, isTyping: false, displayContent: fullContent }
            : msg
        ));
      }
    };

    typeNextWord();
  };

  const handleFollowUpQuestion = async (question: string) => {
    if (!chatId) {
      toast({
        title: "Error",
        description: "Chat session not initialized.",
        variant: "destructive"
      });
      return;
    }
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: question
    };
    const aiMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      type: 'ai',
      content: '',
      isLoading: true
    };

    setMessages(prev => [...prev, userMessage, aiMessage]);
    setIsLoadingNewMessage(true);

    // Auto-scroll to new message immediately
    setTimeout(() => {
      chatScrollRef.current?.scrollTo({
        top: chatScrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
      sourcesScrollRef.current?.scrollTo({
        top: sourcesScrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }, 100);

    try {
      // Ask the follow-up question
      const response = await api.askQuery(chatId, question);

      // Update the AI message with response
      setMessages(prev => prev.map(msg =>
        msg.id === aiMessage.id
          ? { ...msg, content: response, isLoading: false, displayContent: '' }
          : msg
      ));

      // Parse and add new sources from response
      console.log('Extracting sources from follow-up response:', response.substring(0, 500) + '...');
      const extractedSources = extractSourcesFromResponse(response);
      console.log('Extracted follow-up sources:', extractedSources);

      if (extractedSources.length > 0) {
        setSources(prev => [...prev, ...extractedSources]);
        // Animate new sources appearing
        const startIndex = sources.length;
        extractedSources.forEach((_, index) => {
          setTimeout(() => {
            setVisibleSources(prev => new Set([...prev, startIndex + index]));
          }, index * 200);
        });
      }

      // Start typing animation
      typeMessage(aiMessage.id, response);

      // Auto-scroll again after sources are added
      setTimeout(() => {
        chatScrollRef.current?.scrollTo({
          top: chatScrollRef.current.scrollHeight,
          behavior: 'smooth'
        });
        sourcesScrollRef.current?.scrollTo({
          top: sourcesScrollRef.current.scrollHeight,
          behavior: 'smooth'
        });
      }, 100);
    } catch (error) {
      console.error('Error processing follow-up question:', error);
      console.warn('🔄 API unavailable - falling back to mock data for development');

      // Fallback to mock follow-up responses
      setTimeout(() => {
        const mockResponse = mockFollowUpResponses[Math.floor(Math.random() * mockFollowUpResponses.length)];

        setMessages(prev => prev.map(msg =>
          msg.id === aiMessage.id
            ? { ...msg, content: mockResponse, isLoading: false, displayContent: '' }
            : msg
        ));

        // Add some mock sources for the follow-up
        const additionalMockSources = mockSourceCards.slice(-2); // Get last 2 sources
        setSources(prev => [...prev, ...additionalMockSources]);
        animateSourcesLoading(additionalMockSources);

        // Start typing animation
        typeMessage(aiMessage.id, mockResponse);

        // Auto-scroll after mock response
        setTimeout(() => {
          chatScrollRef.current?.scrollTo({
            top: chatScrollRef.current.scrollHeight,
            behavior: 'smooth'
          });
          sourcesScrollRef.current?.scrollTo({
            top: sourcesScrollRef.current.scrollHeight,
            behavior: 'smooth'
          });
        }, 100);
      }, 1200); // Simulate API delay
    } finally {
      setIsLoadingNewMessage(false);
    }
  };

  // Extract sources from markdown response
  const extractSourcesFromResponse = (response: string): SourceCard[] => {
    const sources: SourceCard[] = [];
    const seenUrls = new Set<string>();

    // Extract citations in format [text](url)
    const citationRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const matches = Array.from(response.matchAll(citationRegex));

    console.log('Citation regex matches found:', matches.length);
    console.log('First few matches:', matches.slice(0, 3));

    for (const match of matches) {
      const text = match[1];
      const url = match[2];

      // Skip duplicates
      if (seenUrls.has(url)) continue;
      seenUrls.add(url);

      // Find context around the link (get the sentence/paragraph it's in)
      const matchIndex = match.index || 0;
      const beforeText = response.substring(Math.max(0, matchIndex - 100), matchIndex);
      const afterText = response.substring(matchIndex + match[0].length, matchIndex + match[0].length + 100);

      // Try to find the sentence containing this link
      const contextMatch = (beforeText + match[0] + afterText).match(/[^.!?]*\[([^\]]+)\]\(([^)]+)\)[^.!?]*/);
      const context = contextMatch ? contextMatch[0].trim() : text;

      // Determine source type from URL
      let sourceType = 'Website';
      if (url.includes('.pdf')) sourceType = 'PDF Document';
      else if (url.includes('alphadog')) sourceType = 'Policy Document';
      else if (url.includes('medicare')) sourceType = 'Medicare Plan';

      // Create source card from citation
      sources.push({
        title: url, // Use URL as title instead of link text
        url: url,
        snippet: context,
        type: sourceType
      });

      // Limit to prevent too many sources
      if (sources.length >= 15) break;
    }

    return sources;
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
  };

  const handleLinkClick = (url: string) => {
    // Find the source with matching URL
    const sourceIndex = sources.findIndex(source => source.url === url);

    if (sourceIndex !== -1) {
      setHighlightedSource(sourceIndex);
      // Expand the source card if it's collapsed
      setCollapsedSources(prev => {
        const newSet = new Set(prev);
        newSet.delete(sourceIndex); // Remove from collapsed set to expand it
        return newSet;
      });
      // Scroll to the source element
      const sourceElement = document.querySelector(`[data-source-index="${sourceIndex}"]`);
      if (sourceElement && sourcesScrollRef.current) {
        sourceElement.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }
      // Auto-clear highlight after 3 seconds
      setTimeout(() => {
        setHighlightedSource(null);
      }, 3000);
    }
  };

  const mockAIResponse = `Based on the provided information, the **AARP Medicare Advantage Patriot No Rx MA-MA01 (PPO)** offers comprehensive benefits while having some important exclusions.

## Key Benefits Offered

### **Medical Coverage**
- **$0 monthly premium** with a [$60 Part B premium reduction](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049)
- **No medical deductible** in or out-of-network
- **Maximum out-of-pocket limits**: [$6,700 in-network, $10,100 combined](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049)

### **Doctor Visits & Care**
- [**Primary care visits: $0 copay**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) in-network, $20 out-of-network
- [**Specialist visits: $45 copay**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) in-network, $75 out-of-network
- [**Virtual visits: $0 copay**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) with network providers
- [**Preventive services: $0 copay**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) for covered services

### **Dental Benefits**
- [**$2,000 dental allowance**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) for covered services like cleanings, fillings and crowns
- **$0 copay** for covered network preventive services such as oral exams, routine cleanings, X-rays and fluoride
- **50% coinsurance** for bridges and dentures, $0 copay for other comprehensive services

### **Vision Benefits**
- [**$250 allowance for eyewear**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) every two years
- **$0 copay** for routine eye exams (1 per year)
- Coverage for frames and lenses through network providers

### **Hearing Benefits**
- **$0 copay** for routine hearing exams (1 per year)
- [**Hearing aids from $99-$1,249 copay**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) for OTC and prescription hearing aids (up to 2 per year)

### **Additional Benefits**
- [**$40 OTC credit**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) every quarter for over-the-counter products
- [**Renew Active fitness program: $0 copay**](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) including gym membership and online classes
- [**Meal benefit: $0 copay**](https://www.uhc.com/medicare/alphadog/AAMA25LP0238760_000) for 28 home-delivered meals after hospitalization
- [**Rewards program**: Up to $155 annually](https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049) for wellness activities

## Important Things NOT Covered

### **Transportation & Travel**
- [**Routine transportation**](https://www.uhc.com/medicare/alphadog/AAMA25LP0240580_002) is not covered
- [**Pre-scheduled, pre-planned treatments and elective procedures**](https://www.uhc.com/medicare/alphadog/AAMA25LP0244206_001) outside the U.S. are not covered
- [**Transportation back to the United States**](https://www.uhc.com/medicare/alphadog/AAMA25LP0244206_001) from another country is not covered

### **Medical Services**
- [**Services provided by a dentist**](https://www.uhc.com/medicare/alphadog/AAMA25LP0244206_001) during emergency care are not covered
- [**Provider access fees, appointment fees and administrative fees**](https://www.uhc.com/medicare/alphadog/AAMA25LP0244206_001) are not covered
- Services not covered by Original Medicare are generally excluded

### **Out-of-Network Limitations**
- [**Out-of-network providers have no obligation to treat members**](https://www.uhc.com/medicare/alphadog/AAMA25LP0253194_012) except in emergencies
- Higher cost-sharing applies for out-of-network services
- Some services may require higher copayments or coinsurance when using non-contracted providers

### **Other Exclusions**
- [**Insulin and syringes**](https://www.uhc.com/medicare/alphadog/AAMA25LP0244206_001) are not covered under the diabetes monitoring supplies benefit
- Various Medicare-standard exclusions apply as outlined in the Evidence of Coverage

The plan provides comprehensive coverage with extensive additional benefits beyond Original Medicare, but members should be aware of the limitations, particularly regarding out-of-network care and certain specialized services.`;

  const mockSourceCards = [
    {
      title: "UnitedHealthcare Policy Document",
      url: "policy-doc-1.pdf",
      snippet: "Section 4.2: Coverage Overview\n\nYour plan includes comprehensive medical coverage with preventive care at 100% when using in-network providers. Primary care visits require a $25 copay per visit...",
      type: "PDF Document"
    },
    {
      title: "Benefits Summary",
      url: "benefits-summary.pdf", 
      snippet: "Deductibles and Copays\n\nAnnual Deductible: $1,500 individual / $3,000 family\nOut-of-pocket Maximum: $6,000 individual / $12,000 family\n\nCopayments:\n- Primary Care: $25\n- Specialist: $50\n- Urgent Care: $75",
      type: "Benefits Document"
    },
    {
      title: "Network Provider Directory",
      url: "provider-directory.html",
      snippet: "In-Network Providers\n\nTo maximize your benefits and minimize out-of-pocket costs, always use in-network healthcare providers. Your network includes over 1.2 million physicians and healthcare professionals...",
      type: "Provider Directory"
    },
    {
      title: "Prescription Drug Formulary",
      url: "drug-formulary.pdf",
      snippet: "Tier 1 Medications (Generic): $10 copay\nTier 2 Medications (Preferred Brand): $30 copay\nTier 3 Medications (Non-Preferred Brand): $60 copay\nTier 4 Medications (Specialty): 25% coinsurance",
      type: "Formulary Document"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header onShare={handleShare} />
      
      <div className="flex h-[calc(100vh-4rem)]" ref={containerRef}>
        {/* Left Chat Section */}
        <div 
          className="border-r border-border flex flex-col"
          style={{ width: `${dividerPosition}%` }}
        >
          <div 
            className="flex-1 p-6 scrollable" 
            ref={chatScrollRef}
          >
            <div className="p-4">
              <h2 className="text-base font-semibold text-foreground">
                <Link to="/?view=summary" className="inline-flex items-center hover:opacity-80 transition-all duration-200 group">
                  <ArrowLeft className="h-4 w-4 mr-2 group-hover:-translate-x-1 transition-transform duration-200" />
                  Back to Summary
                </Link>
              </h2>
            </div>

            <div className="space-y-6 pb-32">
              {messages.length > 0 ? (
                messages.map((message) => (
                  <div key={message.id} className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                      message.type === 'user' 
                        ? 'bg-primary' 
                        : 'bg-secondary'
                    }`}>
                      {message.type === 'user' ? (
                        <User className="w-4 h-4 text-primary-foreground" />
                      ) : (
                        <Bot className="w-4 h-4 text-secondary-foreground" />
                      )}
                    </div>
                    <div className="flex-1">
                      {message.type === 'user' ? (
                        <div className="bg-muted rounded-lg p-4">
                          <p className="text-foreground">{message.content}</p>
                        </div>
                      ) : message.isLoading ? (
                        <div className="space-y-2">
                          <Skeleton className="h-4 w-full" />
                          <Skeleton className="h-4 w-3/4" />
                          <Skeleton className="h-4 w-1/2" />
                        </div>
                      ) : (
                        <div className="bg-card border border-border rounded-lg p-4">
                          <div className="prose prose-sm max-w-none text-foreground">
                            <ReactMarkdown
                              components={{
                                a: ({ href, children }) => {
                                  // Handle numbered citations like [1], [2], etc.
                                  const citationMatch = children?.toString().match(/^\[(\d+)\]$/);
                                  if (citationMatch) {
                                    const sourceIndex = parseInt(citationMatch[1]) - 1;
                                    return (
                                      <button
                                        onClick={() => handleCitationClick(sourceIndex)}
                                        className="text-primary hover:text-primary/80 underline underline-offset-2 font-medium cursor-pointer bg-transparent border-none p-0 mx-1"
                                      >
                                        {children}
                                      </button>
                                    );
                                  }
                                  // Handle regular markdown links - these will become sources with distinct blue styling
                                  return (
                                    <button
                                      onClick={() => handleLinkClick(href || '')}
                                      className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline decoration-2 underline-offset-2 font-medium cursor-pointer transition-colors bg-transparent border-none p-0"
                                    >
                                      {children}
                                    </button>
                                  );
                                },
                                p: ({ children }) => (
                                  <p className="text-foreground leading-7 mb-4">
                                    {children}
                                  </p>
                                ),
                                h1: ({ children }) => (
                                  <h1 className="text-xl font-bold text-foreground mt-6 mb-4">
                                    {children}
                                  </h1>
                                ),
                                h2: ({ children }) => (
                                  <h2 className="text-lg font-semibold text-foreground mt-5 mb-3">
                                    {children}
                                  </h2>
                                ),
                                h3: ({ children }) => (
                                  <h3 className="text-base font-semibold text-foreground mt-4 mb-2">
                                    {children}
                                  </h3>
                                ),
                                ul: ({ children }) => (
                                  <ul className="list-disc list-inside space-y-2 mb-4 text-foreground">
                                    {children}
                                  </ul>
                                ),
                                li: ({ children }) => (
                                  <li className="text-foreground">
                                    {children}
                                  </li>
                                ),
                                strong: ({ children }) => (
                                  <strong className="font-semibold text-foreground">
                                    {children}
                                  </strong>
                                )
                              }}
                            >
                              {message.displayContent || message.content}
                            </ReactMarkdown>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center text-muted-foreground py-12">
                  <Bot className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p className="text-lg mb-2">Ask me anything about your insurance</p>
                  <p className="text-sm">I'll provide detailed answers with relevant sources</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Draggable Divider */}
        <div
          className={`w-1 bg-border hover:bg-primary/50 cursor-col-resize transition-colors duration-200 ${
            isDragging ? 'bg-primary' : ''
          }`}
          onMouseDown={handleMouseDown}
        />

        {/* Right Sources Section */}
        <div 
          className="flex flex-col"
          style={{ width: `${100 - dividerPosition}%` }}
        >
          <div className="p-4">
            <h2 className="text-2xl font-semibold text-foreground mt-2">Sources</h2>
          </div>
          
          <div
            className="flex-1 px-4 pb-4 pt-2 scrollable"
            ref={sourcesScrollRef}
          >
            {sources.length > 0 ? (
              <div className="space-y-4 pb-32">
                {sources.map((source, index) => {
                  const isCollapsed = collapsedSources.has(index);
                  const isHighlighted = highlightedSource === index;
                  const isVisible = visibleSources.has(index);
                  return (
                    <div 
                      key={index} 
                      data-source-index={index}
                      className={`bg-card border rounded-lg overflow-hidden transition-all duration-500 transform ${
                        isVisible 
                          ? 'opacity-100 translate-y-0' 
                          : 'opacity-0 translate-y-4'
                      } ${
                        isHighlighted 
                          ? 'border-primary bg-primary/5 shadow-lg' 
                          : 'border-border'
                      }`}
                      style={{
                        transitionDelay: isVisible ? '0ms' : `${index * 200}ms`
                      }}
                    >
                      <div className="p-4 border-b border-border bg-muted/50">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-6 w-6 p-0 hover:bg-muted"
                              onClick={() => toggleSourceCollapse(index)}
                            >
                              {isCollapsed ? (
                                <ChevronDown className="h-3 w-3" />
                              ) : (
                                <ChevronUp className="h-3 w-3" />
                              )}
                            </Button>
                            <h3 className="font-medium text-foreground truncate">{new URL(source.url).hostname}{new URL(source.url).pathname}</h3>
                          </div>
                          <span className="text-xs text-muted-foreground px-2 py-1 bg-background rounded">
                            {source.type}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1 ml-8">Source link</p>
                      </div>
                      <div
                        className={`overflow-hidden transition-all ease-in-out ${
                          isCollapsed ? 'max-h-0 duration-[700ms]' : 'max-h-[600px] duration-700'
                        }`}
                      >
                        <div className="p-2">
                          <div className="bg-muted/30 rounded border overflow-hidden relative">
                            <div className="absolute inset-0 flex items-center justify-center bg-muted/50 z-10 iframe-loading">
                              <div className="flex items-center gap-2 text-muted-foreground text-sm">
                                <div className="animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full"></div>
                                Loading website...
                              </div>
                            </div>
                            <iframe
                              src={source.url}
                              title={`Source: ${new URL(source.url).hostname}`}
                              className="w-full h-[500px] border-0"
                              sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
                              loading="lazy"
                              onLoad={(e) => {
                                // Hide loading overlay when iframe loads successfully
                                const iframe = e.target as HTMLIFrameElement;
                                const loadingOverlay = iframe.parentNode?.querySelector('.iframe-loading') as HTMLElement;
                                if (loadingOverlay) {
                                  loadingOverlay.style.display = 'none';
                                }
                              }}
                              onError={(e) => {
                                // Fallback: show minimal error with external link
                                const iframe = e.target as HTMLIFrameElement;
                                const container = iframe.parentNode as HTMLElement;
                                container.innerHTML = `
                                  <div class="flex items-center justify-center h-[500px] bg-muted/20 rounded">
                                    <div class="text-center p-6">
                                      <div class="w-12 h-12 mx-auto mb-3 bg-orange-100 dark:bg-orange-900 rounded-lg flex items-center justify-center">
                                        <svg class="w-6 h-6 text-orange-600 dark:text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z" />
                                        </svg>
                                      </div>
                                      <p class="text-muted-foreground mb-4 text-sm">This website cannot be embedded</p>
                                      <a href="${source.url}" target="_blank" rel="noopener noreferrer"
                                         class="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors">
                                        <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                        </svg>
                                        Open in New Tab
                                      </a>
                                    </div>
                                  </div>
                                `;
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                {isLoadingNewMessage && (
                  <div className="space-y-4">
                    {[1, 2, 3, 4].map((i) => (
                      <div key={`loading-${i}`} className="bg-card border border-border rounded-lg p-4">
                        <Skeleton className="h-4 w-2/3 mb-2" />
                        <Skeleton className="h-3 w-1/3 mb-4" />
                        <Skeleton className="h-24 w-full" />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground py-12">
                <div className="w-16 h-16 mx-auto mb-4 bg-muted/50 rounded-lg flex items-center justify-center">
                  <svg className="w-8 h-8 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <p className="text-lg mb-2">No sources yet</p>
                <p className="text-sm">Ask a question to see relevant policy documents</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <ChatBar onFollowUpQuestion={handleFollowUpQuestion} insuranceUrl={insuranceUrl} />
    </div>
  );
};

export default Chat;