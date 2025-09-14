// Cookie utility functions for theme persistence
export const getCookie = (name: string): string | null => {
  if (typeof document === 'undefined') return null;
  
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) {
    return parts.pop()?.split(';').shift() || null;
  }
  return null;
};

export const setCookie = (name: string, value: string, days: number = 365): void => {
  if (typeof document === 'undefined') return;
  
  const expires = new Date();
  expires.setTime(expires.getTime() + days * 24 * 60 * 60 * 1000);
  
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
};

export const getThemeFromCookie = (): 'light' | 'dark' | null => {
  const theme = getCookie('theme');
  return theme === 'light' || theme === 'dark' ? theme : null;
};

export const saveThemeToCookie = (theme: 'light' | 'dark'): void => {
  setCookie('theme', theme);
};
