import { Calendar as CalendarIcon } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
} from "./DropdownMenu";
import { Calendar } from "./Calendar";
import { format } from "date-fns";

export function FinishByDateField({
  dueDate,
  setDueDate,
}: {
  dueDate: string;
  setDueDate: (iso: string) => void;
}) {
  const parseLocalDate = (isoDate: string): Date => {
    const [year, month, day] = isoDate.split("-").map(Number);
    return new Date(year, month - 1, day); // No timezone offset
  };

  const selected = dueDate ? parseLocalDate(dueDate) : undefined;

  return (
    <section className="relative">
      <label className="block text-sm font-medium mb-2">Finish By Date</label>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            className="
              w-full flex items-center justify-between 
              px-4 py-3 border rounded-lg bg-background 
              focus:ring-2 focus:ring-primary
            "
          >
            <span
              className={`text-[1rem] ${
                selected ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              {selected ? format(selected, "MM/dd/yyyy") : "Select a date…"}
            </span>
            <CalendarIcon className="w-5 h-5 text-primary" />
          </button>
        </DropdownMenuTrigger>

        <DropdownMenuContent sideOffset={4} align="start" className="p-0">
          <div className="bg-card rounded-lg shadow-lg">
            <Calendar
              mode="single"
              selected={selected}
              onSelect={(date) => {
                if (date) {
                  setDueDate(format(date, "yyyy-MM-dd"));
                }
              }}
            />
            <button
              type="button"
              onClick={() => setDueDate("")}
              className="w-full text-sm text-destructive py-2 border-t hover:bg-muted"
            >
              Clear Date
            </button>
          </div>
        </DropdownMenuContent>
      </DropdownMenu>
    </section>
  );
}
