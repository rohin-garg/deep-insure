import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";

interface ChatBarProps {
  onFollowUpQuestion?: (question: string) => void;
  insuranceUrl?: string;
}

export const ChatBar = ({ onFollowUpQuestion, insuranceUrl }: ChatBarProps) => {
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const isOnChatPage = location.pathname === '/chat';

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      if (isOnChatPage && onFollowUpQuestion) {
        // Handle follow-up question on chat page
        onFollowUpQuestion(query.trim());
        setQuery(""); // Clear the input
      } else if (insuranceUrl) {
        // Navigate to chat page with new question and insurance URL
        navigate(`/chat?q=${encodeURIComponent(query.trim())}&url=${encodeURIComponent(insuranceUrl)}`);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      handleSubmit(e);
    }
  };

  return (
    <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50 w-full max-w-2xl px-4">
      <form onSubmit={handleSubmit} className="relative">
        <div className={`bg-[#e8e8e7]/90 dark:bg-[#201c1c]/90 backdrop-blur-sm border border-border rounded-xl shadow-lg transition-all duration-300 ease-out ${
          isFocused ? 'p-4' : 'p-2'
        }`}>
          <div className="flex items-center gap-3">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder={isOnChatPage ? "Ask a follow up question..." : "Ask about your insurance plan..."}
              className={`border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-xl placeholder:text-muted-foreground/70 transition-all duration-300 ease-out ${
                isFocused ? 'h-12' : 'h-8'
              }`}
            />
            <Button
              type="submit"
              size="sm"
              disabled={!query.trim() || (!isOnChatPage && !insuranceUrl)}
              className="shrink-0 group rainbow-button"
            >
              <Send className="h-4 w-4 transition-transform duration-200 ease-out group-hover:-translate-y-1 group-hover:translate-x-1" />
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
};