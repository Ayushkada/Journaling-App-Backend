import * as React from "react";

const MOBILE_BREAKPOINT = 768;

export const useIsMobile = () => {
  const [isMobile, setIsMobile] = React.useState(() => {
    if (typeof window !== "undefined") {
      return window.innerWidth < MOBILE_BREAKPOINT;
    }
    return false; // Default fallback (e.g., SSR)
  });

  React.useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${MOBILE_BREAKPOINT - 1}px)`);

    const onChange = (event: MediaQueryListEvent) => {
      setIsMobile(event.matches);
    };

    mql.addEventListener("change", onChange);
    setIsMobile(mql.matches); // Set immediately

    return () => {
      mql.removeEventListener("change", onChange);
    };
  }, []);

  return isMobile;
};
