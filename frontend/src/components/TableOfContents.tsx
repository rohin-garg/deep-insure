import { useState, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/utils/utils";

interface TOCItem {
  id: string;
  text: string;
  level: number;
}

interface TableOfContentsProps {
  markdown?: string;
  className?: string;
}

export const TableOfContents = ({ markdown = "", className }: TableOfContentsProps) => {
  const [tocItems, setTocItems] = useState<TOCItem[]>([]);
  const [activeId, setActiveId] = useState<string>("");

  // Extract headers from markdown
  useEffect(() => {
    if (!markdown) {
      setTocItems([]);
      return;
    }

    const headerRegex = /^(#{1,6})\s+(.+)$/gm;
    const items: TOCItem[] = [];
    let match;

    while ((match = headerRegex.exec(markdown)) !== null) {
      const level = match[1].length;
      const text = match[2].trim();
      const id = text
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, '')
        .replace(/\s+/g, '-')
        .trim();
      
      items.push({ id, text, level });
    }

    setTocItems(items);
  }, [markdown]);

  // Handle scroll spy to highlight current section
  useEffect(() => {
    if (tocItems.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      {
        rootMargin: "-20% 0% -35% 0%",
        threshold: 0.1,
      }
    );

    // Observe all heading elements
    tocItems.forEach((item) => {
      const element = document.getElementById(item.id);
      if (element) {
        observer.observe(element);
      }
    });

    return () => observer.disconnect();
  }, [tocItems]);

  const handleClick = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      setActiveId(id);
    }
  };

  if (tocItems.length === 0) {
    return (
      <div className={cn("hidden xl:block w-64 shrink-0", className)}>
        <div className="sticky top-4 p-4">
          <div className="text-sm text-muted-foreground">
            No headings found in this section.
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("hidden xl:block w-64 shrink-0 border-l border-border", className)}>
      <div className="sticky top-4 p-4">
        <h4 className="text-sm font-medium text-foreground mb-3">
          On this page
        </h4>
        <ScrollArea className="h-[calc(100vh-8rem)] scrollable">
          <div className="space-y-1">
            {tocItems.map((item) => (
              <button
                key={item.id}
                onClick={() => handleClick(item.id)}
                className={cn(
                  "block w-full text-left text-sm py-1 px-2 rounded hover:bg-muted/50 transition-colors",
                  "text-muted-foreground hover:text-foreground",
                  activeId === item.id && "text-primary bg-primary/10 hover:bg-primary/15",
                  item.level === 1 && "font-medium",
                  item.level === 2 && "ml-2",
                  item.level === 3 && "ml-4 text-xs",
                  item.level === 4 && "ml-6 text-xs",
                  item.level >= 5 && "ml-8 text-xs"
                )}
              >
                {item.text}
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
};