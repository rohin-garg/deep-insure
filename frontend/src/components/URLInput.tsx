import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FileText, ArrowRight } from "lucide-react";

interface URLInputProps {
  onSubmit: (url: string) => void;
}

export const URLInput = ({ onSubmit }: URLInputProps) => {
  const [url, setUrl] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (url.trim()) {
      onSubmit(url.trim());
    }
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center bg-gradient-to-br from-background to-muted/20">
      <div className="w-full max-w-2xl mx-auto p-8">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary/10 rounded-full mb-4">
            <FileText className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-4xl font-bold text-foreground mb-4">
            Transform Your Insurance Plan
          </h1>
          <p className="text-xl text-muted-foreground mb-2">
            Enter your insurance plan URL to generate an easy-to-navigate wiki
          </p>
          <p className="text-sm text-muted-foreground">
            We'll convert complex documents into a searchable, user-friendly format
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <Input
              type="url"
              placeholder="https://example.com/insurance-plan.pdf"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              className="flex-1 h-12 text-base"
              required
            />
            <Button
              type="submit"
              disabled={!url.trim()}
              className="h-12 px-8 bg-primary hover:bg-primary/90"
            >
              <div className="flex items-center gap-2">
                Generate Wiki
                <ArrowRight className="w-4 h-4" />
              </div>
            </Button>
          </div>
        </form>

        <div className="mt-8 text-center">
          <p className="text-sm text-muted-foreground">
            Don't have a URL? Try our{" "}
            <button
              type="button"
              onClick={() => onSubmit("demo")}
              className="text-primary hover:underline font-medium"
            >
              demo insurance plan
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};