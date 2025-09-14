import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FileText, ArrowRight } from "lucide-react";

interface URLInputProps {
  onSubmit: (url: string) => void;
}

const exampleTexts = [
  "UnitedHealth",
  "Anthem", 
  "Cigna Healthcare",
  "Federal Marketplace",
  "https://www.bcbs.com/explore-affordable-health-plans/individual-family-health-insurance",
  "https://www.uhc.com/medicare/health-plans/details.html/20701/003/H7464011000/2025",
  "https://www.cigna.com/individuals-families/member-guide/plan-documents"
];

export const URLInput = ({ onSubmit }: URLInputProps) => {
  const [url, setUrl] = useState("");
  const [placeholderText, setPlaceholderText] = useState("");
  const [currentTextIndex, setCurrentTextIndex] = useState(0);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Typing animation effect
  useEffect(() => {
    let currentIndex = 0;
    let isDeleting = false;
    let currentText = exampleTexts[currentTextIndex];

    const typeText = () => {
      if (!isDeleting) {
        // Typing phase
        if (currentIndex < currentText.length) {
          setPlaceholderText(currentText.substring(0, currentIndex + 1));
          currentIndex++;
          typingTimeoutRef.current = setTimeout(typeText, 100); // 100ms per character
        } else {
          // Finished typing, pause before deleting
          typingTimeoutRef.current = setTimeout(() => {
            isDeleting = true;
            typeText();
          }, 2000); // 2 second pause
        }
      } else {
        // Deleting phase
        if (currentIndex > 0) {
          setPlaceholderText(currentText.substring(0, currentIndex - 1));
          currentIndex--;
          typingTimeoutRef.current = setTimeout(typeText, 50); // 50ms per character (faster deletion)
        } else {
          // Finished deleting, move to next text
          isDeleting = false;
          setCurrentTextIndex(prev => (prev + 1) % exampleTexts.length);
          currentText = exampleTexts[(currentTextIndex + 1) % exampleTexts.length];
          typeText();
        }
      }
    };

    typeText();

    return () => {
      if (typingTimeoutRef.current) clearTimeout(typingTimeoutRef.current);
    };
  }, [currentTextIndex]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onSubmit(url.trim());
    }
  };

  return (
    <div className="min-h-screen flex items-start justify-center bg-gradient-to-br from-background to-muted/20 pt-32">
      <div className="w-full max-w-2xl mx-auto p-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary/10 rounded-full mb-4">
            <FileText className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Understand Your Insurance Plan
          </h1>
          <p className="text-xl text-muted-foreground mb-2">
            Enter your insurance plan URL or type in a company and an insurance policy to generate a searchable, easy-to-read wiki
          </p>
        </div>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              type="url"
              placeholder={placeholderText}
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="flex-1 h-12 text-base bg-gray-100 dark:bg-gray-800 border-gray-300 dark:border-gray-600"
              required
            />
            <Button
              type="submit"
              disabled={!url.trim()}
              className="h-12 px-8 bg-[#1c398e] hover:bg-[#1e40af] text-white group"
            >
              <div className="flex items-center gap-2">
                Generate Wiki
                <ArrowRight className="w-4 h-4 transition-transform duration-200 group-hover:translate-x-1" />
              </div>
            </Button>
          </div>
        </form>
        <div className="mt-8 text-center">
          <p className="text-sm text-muted-foreground">
            Don't have a URL? Try this {" "}
            <button
              type="button"
              onClick={() => setUrl("https://www.uhc.com/medicare/health-plans/details.html/01054/011/H8768045000/2025?WT.mc_id=8031049")}
              className="text-primary hover:underline font-medium"
            >
              demo policy
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};
