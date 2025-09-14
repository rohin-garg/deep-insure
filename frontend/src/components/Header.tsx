import { Share, Moon, Sun, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { getThemeFromCookie, saveThemeToCookie } from "@/utils/cookies";

interface HeaderProps {
  onShare?: () => void;
  onLogoClick?: () => void;
}

export const Header = ({ onShare, onLogoClick }: HeaderProps) => {
  const [darkMode, setDarkMode] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const navigate = useNavigate();
  const themeButtonRef = useRef<HTMLButtonElement>(null);

  // Load theme from cookie on component mount
  useEffect(() => {
    const savedTheme = getThemeFromCookie();
    if (savedTheme) {
      const isDark = savedTheme === 'dark';
      setDarkMode(isDark);
      if (isDark) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
    } else {
      // Check system preference if no cookie exists
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setDarkMode(prefersDark);
      if (prefersDark) {
        document.documentElement.classList.add('dark');
      }
      saveThemeToCookie(prefersDark ? 'dark' : 'light');
    }
  }, []);

  const createThemeTransition = (isDark: boolean) => {
    // Create the transition element
    const transitionEl = document.createElement('div');
    transitionEl.className = `theme-transition ${isDark ? 'light-to-dark' : 'dark-to-light'}`;
    
    // Position the transition element to cover the full viewport
    transitionEl.style.left = '0px';
    transitionEl.style.top = '0px';
    transitionEl.style.width = '100vw';
    transitionEl.style.height = '100vh';
    
    // Add to document
    document.body.appendChild(transitionEl);
    
    // Add classes to body for smooth transitions
    document.body.classList.add('theme-changing');
    
    // Trigger the animation
    requestAnimationFrame(() => {
      transitionEl.classList.add('active');
    });
    
    // Clean up after animation
    setTimeout(() => {
      document.body.classList.remove('theme-changing');
      document.body.removeChild(transitionEl);
    }, 600);
  };

  const toggleTheme = () => {
    if (isTransitioning) return;
    
    setIsTransitioning(true);
    const newDarkMode = !darkMode;
    
    // Create the visual transition effect
    createThemeTransition(newDarkMode);
    
    // Apply theme change immediately to sync with animation
    setDarkMode(newDarkMode);
    
    if (newDarkMode) {
      document.documentElement.classList.add('dark');
      saveThemeToCookie('dark');
    } else {
      document.documentElement.classList.remove('dark');
      saveThemeToCookie('light');
    }
    
    // Reset transition state after animation completes
    setTimeout(() => {
      setIsTransitioning(false);
    }, 600);
  };

  const handleLogoClick = () => {
    if (onLogoClick) {
      onLogoClick();
    } else {
      navigate('/');
    }
  };

  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    setShowToast(true);

    // Auto-close toast after 1 second
    setTimeout(() => {
      setShowToast(false);
    }, 1000);
  };

  return (
    <>
      <header className="h-16 border-b bg-[#e8e8e7]/90 dark:bg-[#201c1c]/90 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 h-full flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div 
              className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={handleLogoClick}
            >
              <img 
                src="/favicon.ico" 
                alt="DeepInsure Logo" 
                className="h-8 w-8"
              />
              <h1 className="text-2xl font-bold text-primary">
                DeepInsure <span className="text-sm text-muted-foreground font-normal">by <a href="https://www.linkedin.com/in/rohin-garg-362bb8226/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">Rohin</a>, <a href="https://www.linkedin.com/in/jeremy-luu1/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">Jeremy</a>, <a href="https://www.linkedin.com/in/brian-zhang-96b120378/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">Brian</a>, and <a href="https://www.linkedin.com/in/william-zhao-53545b233/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">William</a></span>
              </h1>
            </div>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              ref={themeButtonRef}
              onClick={toggleTheme}
              disabled={isTransitioning}
              aria-label="Toggle theme"
              className="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-200 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 transition-colors duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2"
            >
              <span
                className={`inline-flex h-4 w-4 items-center justify-center transform rounded-full bg-white border border-gray-200 shadow-sm transition-all duration-300 ease-in-out ${
                  darkMode ? 'translate-x-6' : 'translate-x-1'
                }`}
              >
                {darkMode ? (
                  <Moon className="h-2.5 w-2.5 text-gray-600 transition-all duration-300 ease-in-out" />
                ) : (
                  <Sun className="h-2.5 w-2.5 text-yellow-500 transition-all duration-300 ease-in-out" />
                )}
              </span>
            </button>
            
            <Button
              onClick={handleShare}
              size="sm"
              className="bg-primary text-primary-foreground hover:bg-primary/90 group"
            >
              <Share 
                className="h-4 w-4 mr-2 transition-transform duration-200 ease-in-out group-hover:scale-110" 
                style={{
                  animation: 'none'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.animation = 'wobble 0.6s ease-in-out';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.animation = 'none';
                }}
              />
              Share
            </Button>
          </div>
        </div>
      </header>
      
      {/* Toast Notification - Outside header for proper positioning */}
      {showToast && (
        <div 
          className="fixed bottom-4 right-4 z-[100] transform transition-all duration-300 ease-out"
          style={{
            animation: 'slideInFromRight 0.3s ease-out'
          }}
        >
          <div className="bg-background border border-border rounded-lg shadow-lg p-4 flex items-center gap-3 min-w-[200px]">
            <div className="flex-shrink-0">
              <Check className="h-5 w-5 text-green-500" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-foreground">Link Copied!</p>
              <p className="text-xs text-muted-foreground">URL copied to clipboard</p>
            </div>
          </div>
        </div>
      )}
    </>
  );
};