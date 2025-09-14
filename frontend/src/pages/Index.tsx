import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Header } from "@/components/Header";
import { URLInput } from "@/components/URLInput";
import { Navigation } from "@/components/Navigation";
import { ContentArea } from "@/components/ContentArea";
import { TableOfContents } from "@/components/TableOfContents";
import { ChatBar } from "@/components/ChatBar";
import { mockSummary } from "@/utils/mockData";
import { useToast } from "@/hooks/use-toast";

const Index = () => {
  const [searchParams] = useSearchParams();
  const [currentView, setCurrentView] = useState<'input' | 'wiki'>('input');
  const [activeSection, setActiveSection] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  const [currentUrl, setCurrentUrl] = useState('');
  const [isSectionHighlighted, setIsSectionHighlighted] = useState(false);
  const { toast } = useToast();

  // Check URL parameters to determine initial view
  useEffect(() => {
    const viewParam = searchParams.get('view');
    if (viewParam === 'summary') {
      setCurrentView('wiki');
      setActiveSection(mockSummary[0]?.id || '');
    }
  }, [searchParams]);

  const handleUrlSubmit = async (url: string) => {
    setCurrentUrl(url);
    setCurrentView('wiki');
    setLoading(true);
    
    // Simulate API call
    setTimeout(() => {
      setLoading(false);
      setActiveSection(mockSummary[0]?.id || '');
    }, 2000);
  };

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
    const targetSection = mockSummary.find(section => 
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

  const currentSection = mockSummary.find(section => section.id === activeSection);

  return (
    <div className="min-h-screen bg-background">
      <Header onShare={handleShare} onLogoClick={handleLogoClick} />
      
      {currentView === 'input' ? (
        <URLInput onSubmit={handleUrlSubmit} />
      ) : (
        <>
          <div className="flex h-[calc(100vh-4rem)]">
            <Navigation
              sections={mockSummary}
              activeSection={activeSection}
              onSectionClick={handleSectionClick}
              loading={loading}
              className="w-80 hidden lg:flex"
            />
            
            {/* Mobile Navigation */}
            <Navigation
              sections={mockSummary}
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
          loadingUrl={currentUrl}
          isHighlighted={isSectionHighlighted}
        />
            
            <TableOfContents 
              markdown={currentSection?.text}
            />
          </div>
          <ChatBar />
        </>
      )}

    </div>
  );
};

export default Index;
