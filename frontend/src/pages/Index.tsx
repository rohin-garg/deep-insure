import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useSearchParams } from "react-router-dom";
import { useScramble } from "use-scramble";
import { Header } from "@/components/Header";
import { URLInput } from "@/components/URLInput";
import { Navigation } from "@/components/Navigation";
import { ContentArea } from "@/components/ContentArea";
import { TableOfContents } from "@/components/TableOfContents";
import { ChatBar } from "@/components/ChatBar";
import { InsuranceSection, mockSummary } from "@/utils/mockData";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/services/api";

const Index = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [currentView, setCurrentView] = useState<'input' | 'wiki'>('input');
  const [activeSection, setActiveSection] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  const [summary, setSummary] = useState<InsuranceSection[]>([]);
  const [currentInsuranceUrl, setCurrentInsuranceUrl] = useState<string>('');
  const [isSectionHighlighted, setIsSectionHighlighted] = useState(false);
  const { toast } = useToast();
  const loadingRotationRef = useRef(false);

  // Scramble animation for loading text
  const { ref: scrambleRef, replay } = useScramble({
    text: loadingText,
    speed: 0.5,
    tick: 1,
    step: 1,
    scramble: 3,
    seed: 2,
    playOnMount: true,
    onAnimationStart: () => {
      console.log('🎬 Scramble animation started for:', loadingText);
    },
    onAnimationComplete: () => {
      console.log('✅ Scramble animation completed for:', loadingText);
    }
  });

  // Flavor text for loading states
  const loadingMessages = useMemo(() => [
    "Analyzing your insurance policy...",
    "Extracting key coverage details...",
    "Processing policy terms and conditions...",
    "Identifying important benefits and limitations...",
    "Generating comprehensive summary...",
    "Organizing coverage sections...",
    "Preparing interactive navigation...",
    "Almost ready with your personalized wiki..."
  ], []);
  // Check URL parameters to determine initial view
  useEffect(() => {
    const viewParam = searchParams.get('view');
    if (viewParam === 'summary') {
      setCurrentView('wiki');
      setActiveSection(summary[0]?.id || '');
    }
  }, [searchParams, summary]);

  // Debug: Track loadingText changes
  useEffect(() => {
    console.log('📝 loadingText changed to:', loadingText);
  }, [loadingText]);

  const handleUrlSubmit = useCallback(async (url: string) => {
    setCurrentInsuranceUrl(url);
    setCurrentView('wiki');
    setLoading(true);
    loadingRotationRef.current = true;
    console.log('🚀 Starting loading, setting first message:', loadingMessages[0]);
    setLoadingText(loadingMessages[0]);

    // Rotate through loading messages with scramble animation
    let messageIndex = 0;
    const rotateMessages = () => {
      setTimeout(() => {
        if (loadingRotationRef.current) {
          // Move to next message
          messageIndex = (messageIndex + 1) % loadingMessages.length;
          const newMessage = loadingMessages[messageIndex];
          console.log(`🔄 Rotating to message ${messageIndex}:`, newMessage);
          setLoadingText(newMessage);

          // Trigger scramble animation
          console.log('🎯 Calling replay function, replay available:', !!replay);
          if (replay) {
            replay();
          }

          rotateMessages(); // Schedule next rotation
        } else {
          console.log('❌ Loading stopped, ending rotation');
        }
      }, 2500);
    };

    console.log('⏰ Starting message rotation timer');
    rotateMessages(); // Start the rotation

    try {
      const data = await api.getFullSummary(url);
      loadingRotationRef.current = false; // Stop rotation
      setSummary(data);
      setActiveSection(data[0]?.id || '');
      setLoading(false);

      // Update URL params with the insurance URL
      setSearchParams({ url });
    } catch (error) {
      console.error('Error fetching summary:', error);
      console.warn('🔄 API unavailable - falling back to mock summary for development');

      // Fallback to mock data with realistic timing
      setTimeout(() => {
        loadingRotationRef.current = false; // Stop rotation
        setSummary(mockSummary);
        setActiveSection(mockSummary[0]?.id || '');
        setLoading(false);

        // Still update URL params so navigation works
        setSearchParams({ url });

        toast({
          title: "Development Mode",
          description: "Using mock data - API unavailable. Check console for details.",
          variant: "default"
        });
      }, 10000); // Longer delay to show loading messages
    }
  }, [setSearchParams, toast, loadingMessages, replay]);

  const handleSectionClick = (sectionId: string) => {
    setActiveSection(sectionId);

    // Trigger highlight effect
    setIsSectionHighlighted(true);
    setTimeout(() => {
      setIsSectionHighlighted(false);
    }, 1000); // Highlight for 1 second
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    toast({
      title: "Link Copied",
      description: "The link has been copied to your clipboard.",
    });
  };

  const handleLogoClick = () => {
    setCurrentView('input');
    setActiveSection('');
  };

  const handleCitationClick = (link: string) => {
    // Find section that matches the link
    const targetSection = summary.find(section =>
      link.includes(section.id) || section.id.includes(link.replace('-link', ''))
    );

    if (targetSection) {
      setActiveSection(targetSection.id);
      toast({
        title: "Navigated to Section",
        description: `Jumped to ${targetSection.header}`,
      });
    }
  };

  // Load from URL params on mount
  useEffect(() => {
    const urlParam = searchParams.get('url');

    if (urlParam && !currentInsuranceUrl) {
      handleUrlSubmit(urlParam);
    }
  }, [searchParams, currentInsuranceUrl, handleUrlSubmit]);

  const currentSection = summary.find(section => section.id === activeSection);


  return (
    <div className="min-h-screen bg-background">
      <Header onShare={handleShare} onLogoClick={handleLogoClick} />
      
      {currentView === 'input' ? (
        <URLInput onSubmit={handleUrlSubmit} />
      ) : (
        <>
          <div className="flex h-[calc(100vh-4rem)] min-h-0">
            <Navigation
              sections={summary}
              activeSection={activeSection}
              onSectionClick={handleSectionClick}
              loading={loading}
              className="w-80 hidden lg:flex"
            />

            {/* Mobile Navigation */}
            <Navigation
              sections={summary}
              activeSection={activeSection}
              onSectionClick={handleSectionClick}
              loading={loading}
              className="lg:hidden"
            />
            
        <ContentArea
          section={loading ? null : currentSection}
          loading={loading}
          onCitationClick={handleCitationClick}
          enableTypingAnimation={true}
          loadingUrl={currentInsuranceUrl}
          scrambleRef={scrambleRef}
        />
            
            <TableOfContents
              markdown={currentSection?.text}
            />
          </div>
          <ChatBar insuranceUrl={currentInsuranceUrl} />
        </>
      )}

    </div>
  );
};

export default Index;
