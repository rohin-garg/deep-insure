import { useState, useEffect, useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { useScramble } from "use-scramble";
import { Header } from "@/components/Header";
import { URLInput } from "@/components/URLInput";
import { Navigation } from "@/components/Navigation";
import { ContentArea } from "@/components/ContentArea";
import { TableOfContents } from "@/components/TableOfContents";
import { ChatBar } from "@/components/ChatBar";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { InsuranceSection } from "@/utils/mockData";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/services/api";
import { Loader2, FileText, Search, Brain, Zap } from "lucide-react";

const Index = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [currentView, setCurrentView] = useState<'input' | 'wiki'>('input');
  const [activeSection, setActiveSection] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  const [summary, setSummary] = useState<InsuranceSection[]>([]);
  const [currentInsuranceUrl, setCurrentInsuranceUrl] = useState<string>('');
  const { toast } = useToast();

  // Scramble animation for loading text
  const { ref: scrambleRef } = useScramble({
    text: loadingText,
    speed: 0.6,
    tick: 1,
    step: 1,
    scramble: 8,
    seed: 2,
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

  const handleUrlSubmit = useCallback(async (url: string) => {
    setCurrentInsuranceUrl(url);
    setCurrentView('wiki');
    setLoading(true);
    setLoadingText(loadingMessages[0]);
    
    // Rotate through loading messages with scramble animation
    let messageIndex = 0;
    const messageInterval = setInterval(() => {
      messageIndex = (messageIndex + 1) % loadingMessages.length;
      setLoadingText(loadingMessages[messageIndex]);
    }, 2500);

    try {
      const data = await api.getFullSummary(url);
      clearInterval(messageInterval);
      setSummary(data);
      setActiveSection(data[0]?.id || '');
      setLoading(false);

      // Update URL params with the insurance URL
      setSearchParams({ url });
    } catch (error) {
      console.error('Error fetching summary:', error);
      clearInterval(messageInterval);
      toast({
        title: "Error",
        description: "Failed to fetch insurance plan summary. Please try again.",
        variant: "destructive"
      });
      setLoading(false);
      setCurrentView('input'); // Go back to input view on error
    }
  }, [setSearchParams, toast, loadingMessages]);

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
          <div className="flex h-[calc(100vh-4rem)]">
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
        />
            
            <TableOfContents
              markdown={currentSection?.text}
            />
          </div>
          <ChatBar insuranceUrl={currentInsuranceUrl} />
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
            <DialogDescription asChild>
              <div className="text-center space-y-2">
                <p ref={scrambleRef} className="text-sm text-muted-foreground min-h-[1.25rem]">
                  {loadingText}
                </p>
                <div className="flex items-center justify-center gap-1 text-xs text-muted-foreground">
                  <Brain className="h-3 w-3" />
                  <span>AI-powered analysis in progress</span>
                  <Zap className="h-3 w-3" />
                </div>
              </div>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Index;
