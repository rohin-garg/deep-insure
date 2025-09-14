import { useEffect, useState, useRef, useCallback } from "react";
import { useSearchParams, Link } from "react-router-dom";
import { Header } from "@/components/Header";
import { ArrowLeft, User, Bot, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ChatBar } from "@/components/ChatBar";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { api } from "@/services/api";
import { useToast } from "@/hooks/use-toast";

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
      toast({
        title: "Error",
        description: "Failed to process your question. Please try again.",
        variant: "destructive"
      });
      setMessages(prev => prev.map(msg =>
        msg.id === aiMessage.id
          ? { ...msg, content: 'Sorry, I encountered an error processing your question.', isLoading: false }
          : msg
      ));
    } finally {
      setIsLoadingNewMessage(false);
    }
  }, [setSearchParams, toast]);

  // Initialize chat from URL params
  useEffect(() => {
    let hasInitialized = false;

    const initializeChat = async () => {
      if (hasInitialized) return;
      hasInitialized = true;

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
        }
      }

      // If we have a new question and insurance URL
      if (q && url) {
        console.log('Processing initial question');
        await handleInitialQuestion(q, url, existingChatId);
      }
    };

    initializeChat();
  }, [searchParams, handleInitialQuestion]);

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
      const extractedSources = extractSourcesFromResponse(response);
      setSources(prev => [...prev, ...extractedSources]);

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
      toast({
        title: "Error",
        description: "Failed to process your question. Please try again.",
        variant: "destructive"
      });
      setMessages(prev => prev.map(msg =>
        msg.id === aiMessage.id
          ? { ...msg, content: 'Sorry, I encountered an error processing your question.', isLoading: false }
          : msg
      ));
    } finally {
      setIsLoadingNewMessage(false);
    }
  };

  // Extract sources from markdown response
  const extractSourcesFromResponse = (response: string): SourceCard[] => {
    const sources: SourceCard[] = [];
    // Extract citations in format [text](url)
    const citationRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const matches = response.matchAll(citationRegex);

    let index = 0;
    for (const match of matches) {
      const text = match[1];
      const url = match[2];

      // Create source card from citation
      sources.push({
        title: text,
        url: url,
        snippet: `Citation from response: "${text}"`,
        type: 'Citation'
      });
      index++;

      // Limit to prevent too many sources
      if (index >= 10) break;
    }

    return sources;
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
  };

  const mockAIResponse = `Based on your insurance policy documents, I can provide you with comprehensive information about your coverage.

Your policy includes several key benefits:

**Coverage Highlights:**
- Preventive care is covered at 100% when using in-network providers [1]
- Primary care visits have a $25 copay [2]
- Specialist visits require a $50 copay [2]
- Emergency room visits have a $300 copay (waived if admitted) [1]

**Detailed Benefits:**
Your plan covers a wide range of medical services including hospital stays, outpatient surgery, diagnostic tests, and prescription medications [3]. Mental health services are covered at the same level as medical benefits [1].

For more specific information about deductibles and copays [2], please refer to your policy documents. You can also review information about network providers [3] to ensure you're maximizing your benefits.

**Important Notes:**
- All percentages and copays listed are for in-network providers [3]
- Out-of-network services may have reduced coverage [1]
- Prior authorization may be required for certain procedures [2]

Would you like me to elaborate on any specific aspect of your coverage?`;

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
            <div className="mb-6 pb-4 border-b border-border">
              <Link to="/?view=summary">
                <Button variant="ghost" size="sm" className="mb-4">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Summary
                </Button>
              </Link>
              
              <p className="text-muted-foreground">
                Chat AI Generated Title
              </p>
            </div>

            <div className="space-y-6">
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
                            {(message.displayContent || message.content).split('\n').map((line, index) => (
                              <p key={index} className="mb-2 last:mb-0">
                                {line.includes('[') && line.includes('](') ? (
                                  <>
                                    {line.split(/(\[.*?\]\(.*?\))/g).map((part, i) => {
                                      const linkMatch = part.match(/\[(.*?)\]\((.*?)\)/);
                                      if (linkMatch) {
                                        return (
                                          <span key={i} className="text-primary hover:underline cursor-pointer">
                                            {linkMatch[1]}
                                          </span>
                                        );
                                      }
                                      return part;
                                    })}
                                  </>
                                ) : line.includes('[') && line.includes(']') ? (
                                  <>
                                    {line.split(/(\[\d+\])/g).map((part, i) => {
                                      const citationMatch = part.match(/\[(\d+)\]/);
                                      if (citationMatch) {
                                        const sourceIndex = parseInt(citationMatch[1]) - 1;
                                        return (
                                          <button
                                            key={i}
                                            onClick={() => handleCitationClick(sourceIndex)}
                                            className="text-primary hover:text-primary/80 underline underline-offset-2 font-medium cursor-pointer bg-transparent border-none p-0 mx-1"
                                          >
                                            [{citationMatch[1]}]
                                          </button>
                                        );
                                      }
                                      return part;
                                    })}
                                  </>
                                ) : (
                                  line
                                )}
                              </p>
                            ))}
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
          <div className="p-4 border-b border-border">
            <h2 className="text-lg font-semibold text-foreground">Sources</h2>
          </div>
          
          <div 
            className="flex-1 p-4 scrollable"
          >
            {sources.length > 0 ? (
              <div className="space-y-4">
                {sources.map((source, index) => {
                  const isCollapsed = collapsedSources.has(index);
                  const isHighlighted = highlightedSource === index;
                  return (
                    <div 
                      key={index} 
                      data-source-index={index}
                      className={`bg-card border rounded-lg overflow-hidden transition-all duration-300 ${
                        isHighlighted 
                          ? 'border-primary bg-primary/5 shadow-lg' 
                          : 'border-border'
                      }`}
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
                            <h3 className="font-medium text-foreground">{source.title}</h3>
                          </div>
                          <span className="text-xs text-muted-foreground px-2 py-1 bg-background rounded">
                            {source.type}
                          </span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1 ml-8">{source.url}</p>
                      </div>
                      <div 
                        className={`overflow-hidden transition-all ease-in-out ${
                          isCollapsed ? 'max-h-0 duration-[700ms]' : 'max-h-96 duration-700'
                        }`}
                      >
                        <div className="p-4">
                          <pre className="text-sm text-foreground whitespace-pre-wrap font-mono bg-muted/30 p-3 rounded border">
{source.snippet}
                          </pre>
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