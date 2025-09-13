import { useState } from "react";
import { Search, Menu, X } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { InsuranceSection } from "@/lib/mockData";
import { cn } from "@/utils/utils";

interface NavigationProps {
  sections: InsuranceSection[];
  activeSection: string;
  onSectionClick: (sectionId: string) => void;
  className?: string;
  loading?: boolean;
}

export const Navigation = ({
  sections,
  activeSection,
  onSectionClick,
  className,
  loading
}: NavigationProps) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [isCollapsed, setIsCollapsed] = useState(false);

  const filteredSections = sections.filter(section =>
    section.header.toLowerCase().includes(searchTerm.toLowerCase()) ||
    section.text.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className={cn("bg-card border-r flex flex-col", className)}>
      {/* Toggle button for mobile */}
      <div className="p-4 border-b lg:hidden">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsCollapsed(!isCollapsed)}
          className="w-full justify-start"
        >
          {isCollapsed ? <Menu className="h-4 w-4" /> : <X className="h-4 w-4" />}
          <span className="ml-2">Navigation</span>
        </Button>
      </div>

      <div className={cn("flex-1 flex flex-col", isCollapsed && "hidden lg:flex")}>
        {/* Search */}
        <div className="p-4 border-b">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search sections..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        {/* Navigation items */}
        {loading ? (
          <div className="flex-1 p-4 space-y-2">
            <Skeleton className="h-4 w-1/2 mb-4" />
            <Skeleton className="h-8 w-3/4" />
            <Skeleton className="h-8 w-2/3" />
            <Skeleton className="h-8 w-3/4" />
            <Skeleton className="h-8 w-1/2" />
          </div>
        ) : (
          <nav className="flex-1 p-4 space-y-2">
          <h3 className="font-semibold text-sm uppercase tracking-wide text-muted-foreground mb-4">
            Plan Sections
          </h3>
          
          {filteredSections.map((section) => (
            <button
              key={section.id}
              onClick={() => onSectionClick(section.id)}
              className={cn(
                "w-full text-left px-3 py-2 rounded-md text-sm transition-colors",
                "hover:bg-muted/50 focus:outline-none focus:bg-muted",
                activeSection === section.id && "bg-primary/10 text-primary font-medium"
              )}
            >
              {section.header}
            </button>
          ))}

            {filteredSections.length === 0 && (
              <p className="text-sm text-muted-foreground px-3 py-2">
                No sections found matching "{searchTerm}"
              </p>
            )}
          </nav>
        )}
      </div>
    </div>
  );
};