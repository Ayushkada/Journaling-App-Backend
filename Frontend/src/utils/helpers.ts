import {
  baseCategoryColors,
  fallbackCategoryColors,
} from "@/config/categoryColors";
import { format, formatDistanceToNowStrict } from "date-fns";

/**
 * Truncates a string to the given length, adding "..." if needed.
 */
export const truncateString = (str: string, length: number): string => {
  if (str.length <= length) return str;
  return `${str.slice(0, length)}...`;
};

/**
 * Debounces a function with a specified wait time.
 */
export const debounce = <T extends (...args: any[]) => any>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

/**
 * Generates a random unique ID using base-36 and timestamp.
 */
export const generateId = (): string => {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
};

/**
 * Returns true if an object has no keys.
 */
export const isEmpty = (obj: Record<string, any>): boolean => {
  return Object.keys(obj).length === 0;
};

/**
 * Returns a human-friendly string if the update time differs from the created time.
 */
export const getUpdatedText = (
  created: string,
  updated?: string
): string | null => {
  if (!updated || updated === created) return null;

  const updatedTime = new Date(updated);
  const createdTime = new Date(created);
  const diffMs = updatedTime.getTime() - createdTime.getTime();
  const diffHours = diffMs / 36e5;

  if (diffHours < 24) {
    return `Updated ${formatDistanceToNowStrict(updatedTime, {
      addSuffix: true,
    })}`;
  } else {
    return `Updated: ${format(updatedTime, "MMMM d, yyyy 'at' hh:mm a")}`;
  }
};

/**
 * Navigates to a route with support for blocking unsaved changes.
 */
export const navigateWithUnsavedCheck = (
  link: string,
  hasUnsavedChanges: boolean,
  navigate: (path: string) => void,
  setPendingNav: (path: string) => void
) => {
  if (hasUnsavedChanges) setPendingNav(link);
  else navigate(link);
};

/**
 * Returns a color class for a given category.
 * Uses base colors first, supports optional fallback map state.
 */
export const getCategoryColor = (
  category: string | null | undefined,
  options?: {
    fallbackMap?: Record<string, string>;
    setFallbackMap?: (map: Record<string, string>) => void;
  }
): string => {
  if (!category) return "bg-gray-100 text-gray-800";

  const normalized = category.toLowerCase();
  if (baseCategoryColors[normalized]) {
    return baseCategoryColors[normalized];
  }

  const { fallbackMap, setFallbackMap } = options || {};

  if (fallbackMap && setFallbackMap && !fallbackMap[normalized]) {
    const nextColor =
      fallbackCategoryColors[
        Object.keys(fallbackMap).length % fallbackCategoryColors.length
      ];
    setFallbackMap({ ...fallbackMap, [normalized]: nextColor });
    return nextColor;
  }

  if (fallbackMap && fallbackMap[normalized]) {
    return fallbackMap[normalized];
  }

  const hash = normalized
    .split("")
    .reduce((acc, char) => acc + char.charCodeAt(0), 0);

  return fallbackCategoryColors[hash % fallbackCategoryColors.length];
};

/**
 * Default static fallback using hashing
 */
export const hasCookie = (name: string): boolean => {
  const match = document.cookie.match(new RegExp("(^| )" + name + "="));
  return !!match;
};
