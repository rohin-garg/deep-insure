import { useState, useEffect, useRef } from 'react';
import { ExternalLink, AlertCircle } from 'lucide-react';

interface EmbeddedWebsiteProps {
  url: string;
  title?: string;
  className?: string;
}

export const EmbeddedWebsite = ({ url, title, className = "w-full h-[500px]" }: EmbeddedWebsiteProps) => {
  const [currentServiceIndex, setCurrentServiceIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(false);
  const [isPDF, setIsPDF] = useState(false);
  const [isCheckingPDF, setIsCheckingPDF] = useState(true);

  // PDF detection function
  const checkIfPDF = async (url: string): Promise<boolean> => {
    try {
      // Quick pattern checks first
      if (url.match(/\.pdf$/i)) return true;
      if (url.includes('/alphadog/')) return true; // UHC PDFs
      if (url.match(/\/(pdf|docs|documents)\//i)) return true;

      // Try HEAD request to check content type
      try {
        const response = await fetch(url, {
          method: 'HEAD',
          mode: 'no-cors' // This might not give us headers, but won't CORS error
        });
        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/pdf')) return true;
      } catch {
        // CORS will block this, so we fall back to pattern matching
      }

      // Additional pattern checks for common PDF paths
      if (url.match(/\/[A-Z0-9_-]+_\d{3}$/)) return true; // UHC pattern

      return false;
    } catch {
      return false;
    }
  };

  // Screenshot services to try, ordered by reliability
  const getScreenshotUrl = (originalUrl: string, index: number): string | null => {
    // For images, just show them directly
    if (originalUrl.match(/\.(png|jpg|jpeg|gif|webp)$/i)) {
      return originalUrl;
    }

    const screenshotServices = [
      // ScreenshotAPI.net - reliable, high quality (no signup needed)
      () => `https://shot.screenshotapi.net/screenshot?url=${encodeURIComponent(originalUrl)}&width=1024&height=768&output=image&file_type=png&wait_for_event=load`,

      // Thum.io - simple and fast
      () => `https://image.thum.io/get/width/1024/crop/768/noanimate/${originalUrl}`,

      // HTMLCSStoImage - good quality
      () => `https://hcti.io/v1/image?url=${encodeURIComponent(originalUrl)}&selector=body&device_scale=1&viewport_width=1024&viewport_height=768`,

      // PagePeeker - reliable old service
      () => `https://api.pagepeeker.com/v2/thumbs.php?size=l&url=${encodeURIComponent(originalUrl)}`,

      // ScreenshotLayer - Apilayer service
      () => `https://api.screenshotlayer.com/api/capture?access_key=demo&url=${encodeURIComponent(originalUrl)}&viewport=1024x768&width=300`,

      // Mini.screenshot - free service
      () => `https://mini.s-shot.ru/1024x768/PNG/1024/Z100/?${originalUrl}`,

      // Microlink as last resort
      () => `https://api.microlink.io/?url=${encodeURIComponent(originalUrl)}&screenshot=true&embed=screenshot.url&viewport.width=1024&viewport.height=768`,
    ];

    if (index < screenshotServices.length) {
      return screenshotServices[index]();
    }
    return null;
  };

  const handleImageError = () => {
    console.log(`Screenshot service ${currentServiceIndex + 1} failed for ${url}`);

    // Try next screenshot service
    const nextIndex = currentServiceIndex + 1;
    const nextUrl = getScreenshotUrl(url, nextIndex);

    if (nextUrl) {
      setCurrentServiceIndex(nextIndex);
      setIsLoading(true);
    } else {
      // All screenshot services failed
      setError(true);
      setIsLoading(false);
    }
  };

  const handleImageLoad = () => {
    setIsLoading(false);
    setError(false);
  };

  useEffect(() => {
    // Reset state when URL changes
    setCurrentServiceIndex(0);
    setIsLoading(true);
    setError(false);
    setIsCheckingPDF(true);
    setIsPDF(false);

    // Check if URL is a PDF
    checkIfPDF(url).then((isPDF) => {
      setIsPDF(isPDF);
      setIsCheckingPDF(false);
      if (!isPDF) {
        // Only start screenshot loading if not PDF
        setIsLoading(true);
      }
    });
  }, [url]);

  if (error) {
    return (
      <div className="flex items-center justify-center h-[500px] bg-muted/20 rounded border">
        <div className="text-center p-6">
          <div className="w-12 h-12 mx-auto mb-3 bg-orange-100 dark:bg-orange-900 rounded-lg flex items-center justify-center">
            <AlertCircle className="w-6 h-6 text-orange-600 dark:text-orange-400" />
          </div>
          <p className="text-muted-foreground mb-2 text-sm font-medium">Unable to embed this website</p>
          <p className="text-muted-foreground mb-4 text-xs">
            Some websites restrict embedding for security reasons
          </p>
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground text-sm font-medium rounded-lg transition-colors"
          >
            <ExternalLink className="w-4 h-4" />
            Open in New Tab
          </a>
        </div>
      </div>
    );
  }

  // Show loading while checking if PDF
  if (isCheckingPDF) {
    return (
      <div className="relative bg-muted/30 rounded-lg border overflow-hidden">
        <div className="flex items-center justify-center h-[500px]">
          <div className="flex flex-col items-center gap-3">
            <div className="animate-spin w-8 h-8 border-3 border-primary border-t-transparent rounded-full"></div>
            <p className="text-sm text-muted-foreground">Analyzing content...</p>
          </div>
        </div>
      </div>
    );
  }

  // If it's a PDF, show PDF viewer
  if (isPDF) {
    return (
      <div className="relative bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-950 dark:to-blue-900 rounded-lg border overflow-hidden group">
        <div className="relative">
          {/* PDF.js viewer */}
          <iframe
            src={`https://mozilla.github.io/pdf.js/web/viewer.html?file=${encodeURIComponent(url)}`}
            title={`PDF: ${title || new URL(url).hostname}`}
            className="w-full h-[500px] border-0"
            onLoad={() => setIsLoading(false)}
            onError={() => {
              // Fallback to Google Docs viewer
              const iframe = document.querySelector(`iframe[src*="mozilla.github.io"]`) as HTMLIFrameElement;
              if (iframe) {
                iframe.src = `https://docs.google.com/gview?url=${encodeURIComponent(url)}&embedded=true`;
              }
            }}
          />

          {/* PDF indicator badge */}
          <div className="absolute top-4 left-4 bg-red-100 dark:bg-red-900 backdrop-blur-sm rounded-lg px-3 py-1.5">
            <div className="flex items-center gap-2">
              <svg className="w-4 h-4 text-red-600 dark:text-red-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
              </svg>
              <p className="text-xs font-medium text-red-700 dark:text-red-300">PDF Document</p>
            </div>
          </div>

          {/* Download button */}
          <div className="absolute top-4 right-4 bg-background/90 backdrop-blur-sm rounded-lg">
            <a
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium text-primary hover:text-primary/80 transition-colors"
            >
              <ExternalLink className="w-3 h-3" />
              Open PDF
            </a>
          </div>
        </div>
      </div>
    );
  }

  const screenshotUrl = getScreenshotUrl(url, currentServiceIndex);

  return (
    <div className="relative bg-gradient-to-br from-muted/20 to-muted/40 rounded-lg border overflow-hidden group">
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10">
          <div className="flex flex-col items-center gap-3">
            <div className="animate-spin w-8 h-8 border-3 border-primary border-t-transparent rounded-full"></div>
            <p className="text-sm text-muted-foreground">Capturing website...</p>
            {currentServiceIndex > 0 && (
              <p className="text-xs text-muted-foreground">
                Trying service {currentServiceIndex + 1}...
              </p>
            )}
          </div>
        </div>
      )}

      <div className="relative">
        {/* Website screenshot */}
        <img
          key={`${url}-${currentServiceIndex}`} // Force re-render when service changes
          src={screenshotUrl || ''}
          alt={`Preview of ${new URL(url).hostname}`}
          className="w-full h-[500px] object-cover object-top"
          onLoad={handleImageLoad}
          onError={handleImageError}
        />

        {/* Overlay gradient for better text visibility */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

        {/* Interactive overlay */}
        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-3 bg-primary text-primary-foreground rounded-lg shadow-lg hover:bg-primary/90 transition-all transform hover:scale-105"
          >
            <ExternalLink className="w-5 h-5" />
            Open Website
          </a>
        </div>

        {/* Service indicator badge */}
        <div className="absolute top-4 left-4 bg-background/90 backdrop-blur-sm rounded-lg px-3 py-1.5">
          <p className="text-xs font-medium text-muted-foreground">Website Preview</p>
        </div>

        {/* Hostname badge */}
        <div className="absolute bottom-4 left-4 bg-background/90 backdrop-blur-sm rounded-lg px-3 py-1.5">
          <p className="text-xs font-medium text-foreground">{new URL(url).hostname}</p>
        </div>
      </div>
    </div>
  );
};