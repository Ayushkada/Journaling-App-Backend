import { useComboBox } from "@/hooks/useCombobox";
import { useState } from "react";

export function CategoryPicker({
  baseCategories,
  baseCategoryColors,
  fallbackCategoryColors,
  selected,
  setSelected,
}) {
  const { query, setQuery, isOpen, setIsOpen, dropdownRef, filtered, trimmed } =
    useComboBox({
      allOptions: baseCategories,
      selected,
      allowCustom: true,
    });

  const [fallbackMap, setFallbackMap] = useState<Record<string, string>>({});

  const getColorClass = (cat: string, index: number) => {
    if (baseCategoryColors[cat]) return baseCategoryColors[cat];

    if (!fallbackMap[cat]) {
      const fallbackColor =
        fallbackCategoryColors[
          Object.keys(fallbackMap).length % fallbackCategoryColors.length
        ];
      setFallbackMap((prev) => ({ ...prev, [cat]: fallbackColor }));
    }

    return fallbackMap[cat];
  };

  const handleSelect = (val: string) => {
    const value = val.trim();
    if (!value || selected.includes(value)) return;
    setSelected([...selected, value]);
    setQuery("");
    setIsOpen(false);
  };

  return (
    <div ref={dropdownRef}>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {selected.map((cat, index) => {
            const colorClass = getColorClass(cat, index);
            return (
              <span
                key={cat}
                className={`inline-flex items-center text-xs font-medium px-2 py-1 rounded-full ${colorClass}`}
              >
                {cat}
                <button
                  type="button"
                  onClick={() => setSelected(selected.filter((c) => c !== cat))}
                  className="ml-1 text-sm font-bold leading-none focus:outline-none"
                  aria-label={`Remove category ${cat}`}
                >
                  ×
                </button>
              </span>
            );
          })}
        </div>
      )}

      {/* Wrap input + dropdown in relative container */}
      <div className="relative">
        {/* Input */}
        <input
          value={query}
          onFocus={() => setIsOpen(true)}
          onClick={() => setIsOpen(true)}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && trimmed) {
              handleSelect(trimmed);
            }
          }}
          placeholder="Type or select a category"
          className="w-full px-4 py-2 border rounded-lg"
        />

        {/* Floating Dropdown */}
        {isOpen && (filtered.length > 0 || trimmed) && (
          <div className="absolute left-0 top-full z-20 mt-1 w-full border rounded shadow max-h-60 overflow-auto bg-white">
            {filtered.map((opt) => (
              <div
                key={opt}
                onMouseDown={() => handleSelect(opt)}
                className="px-4 py-2 hover:bg-accent cursor-pointer"
              >
                {opt}
              </div>
            ))}
            {!filtered.includes(trimmed) &&
              trimmed &&
              !selected.includes(trimmed) && (
                <div
                  onMouseDown={() => handleSelect(trimmed)}
                  className="px-4 py-2 italic text-muted-foreground hover:bg-accent cursor-pointer"
                >
                  Add "{trimmed}"
                </div>
              )}
          </div>
        )}
      </div>
    </div>
  );
}
