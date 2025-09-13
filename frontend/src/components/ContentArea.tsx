import ReactMarkdown from 'react-markdown';
import { InsuranceSection } from "@/utils/mockData";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertCircle } from "lucide-react";

interface ContentAreaProps {
  section?: InsuranceSection;
  loading?: boolean;
  onCitationClick?: (link: string) => void;
}

export const ContentArea = ({ section, loading, onCitationClick }: ContentAreaProps) => {
  if (loading) {
    return (
      <div className="flex-1 p-8 space-y-4">
        <Skeleton className="h-8 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-2/3" />
        <div className="space-y-2 mt-8">
          <Skeleton className="h-6 w-1/2" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
        </div>
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
    <div className="flex-1 overflow-y-auto">
      <article className="max-w-4xl mx-auto p-8 prose prose-slate dark:prose-invert prose-headings:text-foreground prose-p:text-foreground prose-strong:text-foreground prose-li:text-foreground">
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
          {section.text}
        </ReactMarkdown>
      </article>
    </div>
  );
};