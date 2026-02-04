"use client";

import { useState, useRef, forwardRef, useImperativeHandle, useCallback } from 'react';

export interface BrowserHandle {
  navigate: (url: string) => void;
  getUrl: () => string;
  getTitle: () => string;
  search: (query: string) => void;
  scroll: (direction: 'up' | 'down', amount?: number) => void;
  getSnapshot: () => Promise<string>;
  analyze: () => Promise<any>;
}

interface RealBrowserProps {
  initialUrl?: string;
  onNavigate?: (url: string, title: string) => void;
  onLoad?: (url: string, title: string) => void;
  disabled?: boolean;
}

const HOMEPAGE = 'https://www.google.com/search?igu=1';

const RealBrowser = forwardRef<BrowserHandle, RealBrowserProps>(({
  initialUrl = HOMEPAGE,
  onNavigate,
  onLoad,
  disabled = false,
}, ref) => {
  const [url, setUrl] = useState(initialUrl);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const navigateToUrl = useCallback((targetUrl: string) => {
    let normalizedUrl = targetUrl.trim();
    if (!normalizedUrl.startsWith('http')) normalizedUrl = 'https://' + normalizedUrl;
    
    setUrl(normalizedUrl);
    onNavigate?.(normalizedUrl, 'Browser');
  }, [onNavigate]);

  useImperativeHandle(ref, () => ({
    navigate: (newUrl: string) => navigateToUrl(newUrl),
    getUrl: () => url,
    getTitle: () => 'Browser',
    search: (query: string) => {
      const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}&igu=1`;
      navigateToUrl(searchUrl);
    },
    scroll: (direction: 'up' | 'down', amount = 300) => {
      if (iframeRef.current?.contentWindow) {
        try {
          const scrollAmount = direction === 'down' ? amount : -amount;
          iframeRef.current.contentWindow.scrollBy(0, scrollAmount);
        } catch {
          console.log('Cannot scroll iframe due to CORS');
        }
      }
    },
    getSnapshot: async () => 'Simple Iframe Mode',
    analyze: async () => ({})
  }), [url, navigateToUrl]);

  return (
    <div className="w-full h-full bg-white overflow-hidden">
      <iframe
        ref={iframeRef}
        src={url}
        className="w-full h-full border-0"
        sandbox="allow-same-origin allow-scripts allow-popups allow-forms allow-modals"
        onLoad={() => {
          onLoad?.(url, 'Browser');
        }}
        title="Browser"
      />
    </div>
  );
});

RealBrowser.displayName = 'RealBrowser';

export default RealBrowser;
