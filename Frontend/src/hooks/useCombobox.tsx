// hooks/useComboBox.ts
import { useEffect, useRef, useState } from "react";

export function useComboBox({
  allOptions,
  selected = [],
  allowCustom = false,
}: {
  allOptions: string[];
  selected?: string[];
  allowCustom?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const trimmed = query.trim();

  const filtered = allOptions.filter(
    (opt) =>
      !selected.includes(opt) &&
      opt.toLowerCase().includes(trimmed.toLowerCase())
  );

  const handleClickOutside = (e: MouseEvent) => {
    if (
      dropdownRef.current &&
      !dropdownRef.current.contains(e.target as Node)
    ) {
      setIsOpen(false);
    }
  };

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return {
    query,
    setQuery,
    isOpen,
    setIsOpen,
    dropdownRef,
    filtered,
    trimmed,
  };
}
