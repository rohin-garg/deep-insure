import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { Header } from "@/components/Header";
import { URLInput } from "@/components/URLInput";
import { Navigation } from "@/components/Navigation";
import { ContentArea } from "@/components/ContentArea";
import { TableOfContents } from "@/components/TableOfContents";
import { ChatBar } from "@/components/ChatBar";
import { InsuranceSection } from "@/utils/mockData";
import { useToast } from "@/hooks/use-toast";
import { api } from "@/services/api";

const Index = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [currentView, setCurrentView] = useState<'input' | 'wiki'>('input');
  const [activeSection, setActiveSection] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<InsuranceSection[]>([]);
  const [currentInsuranceUrl, setCurrentInsuranceUrl] = useState<string>('');
  const { toast } = useToast();

  const handleUrlSubmit = async (url: string) => {
    setCurrentView('wiki');
    setLoading(true);
    setCurrentInsuranceUrl(url);

    try {
      const data = await api.getFullSummary(url);
      setSummary(data);
      setActiveSection(data[0]?.id || '');
      setLoading(false);

      // Update URL params with the insurance URL
      setSearchParams({ url });
    } catch (error) {
      console.error('Error fetching summary:', error);
      toast({
        title: "Error",
        description: "Failed to fetch insurance plan summary. Please try again.",
        variant: "destructive"
      });
      setLoading(false);
      setCurrentView('input'); // Go back to input view on error
    }
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
  }, []);

  const currentSection = summary.find(section => section.id === activeSection);

  return (
    <div className="min-h-screen bg-background">
      <Header onShare={handleShare} />
      
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
    </div>
  );
};

export default Index;
