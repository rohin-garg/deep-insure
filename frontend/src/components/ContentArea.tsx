import ReactMarkdown from 'react-markdown';
import { InsuranceSection } from "@/utils/mockData";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, Loader2, Search, Brain, Zap } from "lucide-react";
import { useState, useEffect, useRef } from "react";

interface ContentAreaProps {
  section?: InsuranceSection;
  loading?: boolean;
  onCitationClick?: (link: string) => void;
  enableTypingAnimation?: boolean;
  loadingUrl?: string;
  scrambleRef?: React.RefObject<HTMLElement>;
}

export const ContentArea = ({ section, loading, onCitationClick, enableTypingAnimation = false, loadingUrl = '', scrambleRef }: ContentAreaProps) => {
  const [displayText, setDisplayText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [dotCount, setDotCount] = useState(1);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const dotIntervalRef = useRef<NodeJS.Timeout | null>(null);


  // Word-by-word typing animation
  const typeText = (text: string) => {
    if (!enableTypingAnimation) {
      setDisplayText(text);
      return;
    }

    const words = text.split(' ');
    let currentWordIndex = 0;
    const typingSpeed = 6; // milliseconds per word (4x faster than before)

    const typeNextWord = () => {
      if (currentWordIndex < words.length) {
        const currentText = words.slice(0, currentWordIndex + 1).join(' ');
        setDisplayText(currentText);
        setIsTyping(true);
        currentWordIndex++;
        typingTimeoutRef.current = setTimeout(typeNextWord, typingSpeed);
      } else {
        setIsTyping(false);
      }
    };

    typeNextWord();
  };

  // Simple dot animation for loading
  useEffect(() => {
    if (loading) {
      dotIntervalRef.current = setInterval(() => {
        setDotCount(prev => (prev % 3) + 1);
      }, 500);
    } else {
      if (dotIntervalRef.current) {
        clearInterval(dotIntervalRef.current);
        dotIntervalRef.current = null;
      }
    }

    return () => {
      if (dotIntervalRef.current) {
        clearInterval(dotIntervalRef.current);
        dotIntervalRef.current = null;
      }
    };
  }, [loading]);

  // Start typing animation when section changes
  useEffect(() => {
    if (section && !loading) {
      typeText(section.text);
    }
  }, [section, loading, enableTypingAnimation]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
      if (dotIntervalRef.current) {
        clearInterval(dotIntervalRef.current);
      }
    };
  }, []);
  if (loading) {
    console.log('📺 ContentArea rendering loading state, scrambleRef available:', !!scrambleRef);
    return (
      <div className="flex-1 scrollable">
        <article className="max-w-4xl mx-auto p-8 prose prose-slate dark:prose-invert prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-li:text-foreground">
          <div className="mb-6">
            <h1 className="text-2xl font-bold text-foreground">
              Searching for insurance details{'.'.repeat(dotCount)}
            </h1>
          </div>
          <div className="border-b border-border pb-4 mb-6">
            <p ref={scrambleRef} className="text-lg text-muted-foreground">
            </p>
          </div>
        </article>
      </div>
    );
  }

  if (!section) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="text-center max-w-md">
          <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">
            No Section Selected
          </h3>
          <p className="text-muted-foreground">
            Select a section from the navigation to view its content.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 scrollable">
      <article className="max-w-4xl mx-auto p-8 pb-32 prose prose-slate dark:prose-invert prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-li:text-foreground">
        <ReactMarkdown
          components={{
            a: ({ href, children }) => (
              <button
                onClick={() => onCitationClick?.(href || '')}
                className="text-primary hover:text-primary/80 underline underline-offset-2 font-medium cursor-pointer bg-transparent border-none p-0"
              >
                {children}
              </button>
            ),
            h1: ({ children }) => {
              const id = children?.toString()
                .toLowerCase()
                .replace(/[^a-z0-9\s-]/g, '')
                .replace(/\s+/g, '-')
                .trim() || '';
              return (
                <h1 id={id} className="text-3xl font-bold text-foreground border-b border-border pb-4 mb-6">
                  {children}
                </h1>
              );
            },
            h2: ({ children }) => {
              const id = children?.toString()
                .toLowerCase()
                .replace(/[^a-z0-9\s-]/g, '')
                .replace(/\s+/g, '-')
                .trim() || '';
              return (
                <h2 id={id} className="text-2xl font-semibold text-foreground mt-8 mb-4">
                  {children}
                </h2>
              );
            },
            h3: ({ children }) => {
              const id = children?.toString()
                .toLowerCase()
                .replace(/[^a-z0-9\s-]/g, '')
                .replace(/\s+/g, '-')
                .trim() || '';
              return (
                <h3 id={id} className="text-xl font-semibold text-foreground mt-6 mb-3">
                  {children}
                </h3>
              );
            },
            p: ({ children }) => (
              <p className="text-foreground leading-7 mb-4">
                {children}
              </p>
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
            )
          }}
        >
          {displayText || section.text}
        </ReactMarkdown>
      </article>
    </div>
  );
};