import { useComboBox } from "@/hooks/useCombobox";

export type JournalOption = {
  id: string;
  title: string;
};

export function SearchComboBox({
  items,
  onSelect,
  placeholder = "Search...",
}: {
  items: JournalOption[];
  onSelect: (id: string) => void;
  placeholder?: string;
}) {
  const titles = items.map((item) => item.title);
  const { query, setQuery, isOpen, setIsOpen, dropdownRef, filtered } =
    useComboBox({ allOptions: titles });

  return (
    <div ref={dropdownRef} className="relative">
      <input
        value={query}
        placeholder={placeholder}
        className="w-full px-4 py-2 border rounded-lg"
        onFocus={() => setIsOpen(true)}
        onClick={() => setIsOpen(true)}
        onChange={(e) => setQuery(e.target.value)}
      />

      {isOpen && filtered.length > 0 && (
        <div className="absolute top-full left-0 w-full mt-1 border rounded bg-white z-10 shadow max-h-60 overflow-auto">
          {filtered.map((title) => {
            const item = items.find((j) => j.title === title);
            if (!item) return null;

            return (
              <div
                key={item.id}
                onMouseDown={() => {
                  onSelect(item.id);
                  setQuery("");
                  setIsOpen(false);
                }}
                className="px-4 py-2 hover:bg-accent cursor-pointer"
              >
                <span
                  className="block truncate max-w-full"
                  title={item.title} // Tooltip on hover
                >
                  {item.title}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
