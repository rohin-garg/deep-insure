import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";

interface ChatBarProps {
  onFollowUpQuestion?: (question: string) => void;
}

export const ChatBar = ({ onFollowUpQuestion }: ChatBarProps) => {
  const [query, setQuery] = useState("");
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
      } else {
        // Navigate to chat page with new question
        navigate(`/chat?q=${encodeURIComponent(query.trim())}`);
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
        <div className="bg-[#e8e8e7]/90 dark:bg-[#201c1c]/90 backdrop-blur-sm border border-border rounded-xl shadow-lg p-4">
          <div className="flex items-center gap-3">
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={isOnChatPage ? "Ask a follow up question..." : "Ask about your insurance plan..."}
              className="border-0 bg-transparent focus-visible:ring-0 focus-visible:ring-offset-0 text-xl placeholder:text-muted-foreground/70 h-12"
            />
            <Button
              type="submit"
              size="sm"
              disabled={!query.trim()}
              className="shrink-0"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
};