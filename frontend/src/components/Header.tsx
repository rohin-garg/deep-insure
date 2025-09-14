import { Share, Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
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
  const [showShareDialog, setShowShareDialog] = useState(false);
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
    setShowShareDialog(true);
    
    // Auto-close dialog after 1 second
    setTimeout(() => {
      setShowShareDialog(false);
    }, 1000);
    
    // Call the onShare prop if provided
    if (onShare) {
      onShare();
    }
  };

  return (
    <header className="h-16 border-b bg-[#e8e8e7]/90 dark:bg-[#201c1c]/90 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 h-full flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h1 
            className="text-2xl font-bold text-primary cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleLogoClick}
          >
            DeepInsure <span className="text-sm text-muted-foreground font-normal">by <a href="https://www.linkedin.com/in/rohin-garg-362bb8226/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">Rohin</a>, <a href="https://www.linkedin.com/in/jeremy-luu1/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">Jeremy</a>, <a href="https://www.linkedin.com/in/brian-zhang-96b120378/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">Brian</a>, and <a href="https://www.linkedin.com/in/william-zhao-53545b233/" target="_blank" rel="noopener noreferrer" className="hover:text-primary underline">William</a></span>
          </h1>
        </div>
        
        <div className="flex items-center gap-2">
          <Button
            ref={themeButtonRef}
            variant="ghost"
            size="sm"
            onClick={toggleTheme}
            disabled={isTransitioning}
            aria-label="Toggle theme"
            className="group relative overflow-hidden"
          >
            {/* Sweeping fill background */}
            <div className="absolute inset-0 bg-gradient-to-tr from-muted/30 to-muted/30 transform -translate-x-full -translate-y-full group-hover:translate-x-0 group-hover:translate-y-0 transition-transform duration-300 ease-out"></div>
            
            <div className="relative z-10">
              <Sun className={`h-4 w-4 transition-all duration-300 ease-in-out absolute inset-0 ${
                darkMode 
                  ? 'opacity-100 scale-100 rotate-0' 
                  : 'opacity-0 scale-75 rotate-180'
              }`} />
              <Moon className={`h-4 w-4 transition-all duration-300 ease-in-out ${
                darkMode 
                  ? 'opacity-0 scale-75 -rotate-180' 
                  : 'opacity-100 scale-100 rotate-0'
              }`} />
            </div>
          </Button>
          
          <Button
            onClick={handleShare}
            size="sm"
            className="bg-primary text-primary-foreground hover:bg-primary/90"
          >
            <Share className="h-4 w-4 mr-2" />
            Share
          </Button>
        </div>
      </div>
      
      {/* Share Dialog */}
      <Dialog open={showShareDialog} onOpenChange={() => {}}>
        <DialogContent className="sm:max-w-md text-center [&>button]:hidden">
          <DialogHeader className="text-center">
            <DialogTitle className="flex items-center justify-center gap-2">
              <Share className="h-5 w-5" />
              Link Copied!
            </DialogTitle>
            <DialogDescription className="text-center">
              The page URL has been copied to your clipboard.
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </header>
  );
};