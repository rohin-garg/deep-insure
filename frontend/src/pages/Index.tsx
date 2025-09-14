import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Header } from "@/components/Header";
import { URLInput } from "@/components/URLInput";
import { Navigation } from "@/components/Navigation";
import { ContentArea } from "@/components/ContentArea";
import { TableOfContents } from "@/components/TableOfContents";
import { ChatBar } from "@/components/ChatBar";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { mockSummary } from "@/utils/mockData";
import { useToast } from "@/hooks/use-toast";
import { Loader2, FileText, Search, Brain, Zap } from "lucide-react";

const Index = () => {
  const [searchParams] = useSearchParams();
  const [currentView, setCurrentView] = useState<'input' | 'wiki'>('input');
  const [activeSection, setActiveSection] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  const [currentUrl, setCurrentUrl] = useState('');
  const { toast } = useToast();

  // Flavor text for loading states
  const loadingMessages = [
    "Analyzing your insurance policy...",
    "Extracting key coverage details...",
    "Processing policy terms and conditions...",
    "Identifying important benefits and limitations...",
    "Generating comprehensive summary...",
    "Organizing coverage sections...",
    "Preparing interactive navigation...",
    "Almost ready with your personalized wiki..."
  ];

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
    setLoadingText(loadingMessages[0]);
    
    // Rotate through loading messages
    let messageIndex = 0;
    const messageInterval = setInterval(() => {
      messageIndex = (messageIndex + 1) % loadingMessages.length;
      setLoadingText(loadingMessages[messageIndex]);
    }, 300);
    
    // Simulate API call
    setTimeout(() => {
      clearInterval(messageInterval);
      setLoading(false);
      setActiveSection(mockSummary[0]?.id || '');
    }, 2000);
  };

  const handleSectionClick = (sectionId: string) => {
    setActiveSection(sectionId);
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
        />
            
            <TableOfContents 
              markdown={currentSection?.text}
            />
          </div>
          <ChatBar />
        </>
      )}

      {/* Loading Dialog */}
      <Dialog open={loading} onOpenChange={() => {}}>
        <DialogContent className="sm:max-w-md text-center [&>button]:hidden">
          <DialogHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="relative">
                <Loader2 className="h-12 w-12 animate-spin text-primary" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <FileText className="h-6 w-6 text-primary/70" />
                </div>
              </div>
            </div>
            <DialogTitle className="flex items-center justify-center gap-2 text-lg">
              <Search className="h-5 w-5" />
              Searching for insurance details...
            </DialogTitle>
            <DialogDescription className="text-center space-y-2">
              <p className="text-sm text-muted-foreground">
                {loadingText}
              </p>
              <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground">
                <Brain className="h-3 w-3" />
                <span>AI-powered analysis in progress</span>
                <Zap className="h-3 w-3" />
              </div>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Index;
