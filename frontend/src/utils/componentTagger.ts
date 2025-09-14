import type { Plugin } from 'vite';

export function customComponentTagger(): Plugin {
  return {
    name: 'custom-component-tagger',
    transform(code: string, id: string) {
      // Only process React components (tsx/jsx files)
      if (!id.endsWith('.tsx') && !id.endsWith('.jsx')) {
        return null;
      }

      // Skip node_modules
      if (id.includes('node_modules')) {
        return null;
      }

      // Extract component name from file path
      const fileName = id.split('/').pop()?.replace(/\.(tsx|jsx)$/, '') || 'Unknown';

      // Add data-component attribute to JSX elements
      const transformedCode = code.replace(
        /(<[A-Z][a-zA-Z0-9]*(?:\s[^>]*)?)(>)/g,
        `$1 data-component="${fileName}"$2`
      );

      // Also add to div elements that might be component roots
      const finalCode = transformedCode.replace(
        /(<div(?:\s[^>]*)?)(>)/g,
        (match, p1, p2) => {
          // Only add if it doesn't already have data-component
          if (p1.includes('data-component')) {
            return match;
          }
          return `${p1} data-component="${fileName}"${p2}`;
        }
      );

      return {
        code: finalCode,
        map: null
      };
    }
  };
}