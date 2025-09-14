import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { InsuranceSection } from "@/utils/mockData";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle, Loader2, Search, Brain, Zap } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { useScramble } from "use-scramble";

interface ContentAreaProps {
  section?: InsuranceSection;
  loading?: boolean;
  onCitationClick?: (link: string) => void;
  enableTypingAnimation?: boolean;
  loadingUrl?: string;
}

export const ContentArea = ({ section, loading, onCitationClick, enableTypingAnimation = false, loadingUrl = '' }: ContentAreaProps) => {
  const [displayText, setDisplayText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [loadingText, setLoadingText] = useState('');
  const [dotCount, setDotCount] = useState(1);
  const typingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const loadingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const dotIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Scramble animation for loading text
  const { ref: scrambleRef } = useScramble({
    text: loadingText,
    speed: 0.6,
    tick: 1,
    step: 1,
    scramble: 8,
    seed: 2,
  });


  // Flavor text for loading states with URL
  const loadingMessages = [
    `Scanning ${loadingUrl}...`,
    "Analyzing your insurance policy...",
    "Extracting key coverage details...",
    "Processing policy terms and conditions...",
    "Identifying important benefits and limitations...",
    "Generating comprehensive summary...",
    "Organizing coverage sections...",
    "Preparing interactive navigation...",
    "Almost ready with your personalized wiki..."
  ];

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

  // Start loading text rotation when loading starts
  useEffect(() => {
    if (loading && loadingUrl) {
      setLoadingText(loadingMessages[0]);
      
      // Rotate through loading messages
      let messageIndex = 0;
      loadingIntervalRef.current = setInterval(() => {
        messageIndex = (messageIndex + 1) % loadingMessages.length;
        setLoadingText(loadingMessages[messageIndex]);
      }, 300);

      // Start dot animation
      dotIntervalRef.current = setInterval(() => {
        setDotCount(prev => (prev % 3) + 1);
      }, 500);
    } else {
      if (loadingIntervalRef.current) {
        clearInterval(loadingIntervalRef.current);
      }
      if (dotIntervalRef.current) {
        clearInterval(dotIntervalRef.current);
      }
    }

    return () => {
      if (loadingIntervalRef.current) {
        clearInterval(loadingIntervalRef.current);
      }
      if (dotIntervalRef.current) {
        clearInterval(dotIntervalRef.current);
      }
    };
  }, [loading, loadingUrl]);

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
      if (loadingIntervalRef.current) {
        clearInterval(loadingIntervalRef.current);
      }
      if (dotIntervalRef.current) {
        clearInterval(dotIntervalRef.current);
      }
    };
  }, []);
  if (loading) {
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
          remarkPlugins={[remarkGfm]}
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
            p: ({ children, node }) => {
              // Check if paragraph contains only bold text (potential heading)
              if (node && node.children && node.children.length === 1) {
                const child = node.children[0];

                if (child.type === 'element' && child.tagName === 'strong') {
                  const boldText = child.children?.[0]?.value || '';
                  const id = boldText.toLowerCase()
                    .replace(/[^a-z0-9\s-]/g, '')
                    .replace(/\s+/g, '-')
                    .trim();

                  return (
                    <h4 id={id} className="text-lg font-semibold text-foreground mb-4 mt-6">
                      {boldText}
                    </h4>
                  );
                }
              }

              return (
                <p className="text-foreground leading-7 mb-4">
                  {children}
                </p>
              );
            },
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
            table: ({ children }) => (
              <div className="overflow-x-auto my-6">
                <table className="min-w-full border border-border rounded-lg">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-muted">
                {children}
              </thead>
            ),
            tbody: ({ children }) => (
              <tbody className="divide-y divide-border">
                {children}
              </tbody>
            ),
            tr: ({ children }) => (
              <tr className="hover:bg-muted/50">
                {children}
              </tr>
            ),
            th: ({ children }) => (
              <th className="px-4 py-3 text-left text-sm font-medium text-foreground border-r border-border last:border-r-0">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-4 py-3 text-sm text-foreground border-r border-border last:border-r-0">
                {children}
              </td>
            )
          }}
        >
          {displayText || section.text}
        </ReactMarkdown>
      </article>
    </div>
  );
};