import * as React from "react";
import { useNavigate } from "react-router-dom";
import { Calendar, Edit, Eye, Trash2 } from "lucide-react";
import { cn } from "@/utils/classnames";
import { Button } from "@/components/ui/Button";
import {
  Card as BaseCard,
  CardContent as BaseCardContent,
} from "@/components/ui/Card";

export interface JournalCardProps {
  id: string;
  title: string;
  preview: string;
  date: string;
  image: string | null;
  emojis: string[];
  viewUrl: string;
  editUrl: string;
  onDelete: () => void;
}

export const JournalCard: React.FC<JournalCardProps> = ({
  title,
  preview,
  date,
  image,
  emojis,
  viewUrl,
  editUrl,
  onDelete,
}) => {
  const navigate = useNavigate();

  const handleCardClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;

    // Prevent navigation if the click is on a button or inside one
    if (target.closest("button")) return;

    navigate(viewUrl);
  };

  return (
    <BaseCard
      tabIndex={0}
      role="button"
      onClick={handleCardClick}
      className="cursor-pointer shadow-md border bg-card w-full h-full transition-transform duration-200 hover:scale-[1.02] overflow-hidden"
    >
      <div className="flex flex-col sm:flex-row h-full">
        {image && (
          <div className="hidden lg:block w-1/3 flex-shrink-0 overflow-hidden bg-muted">
            <img
              src={image}
              alt={title}
              className="w-full h-full object-cover object-center"
            />
          </div>
        )}

        <BaseCardContent
          className={cn(
            "flex flex-col justify-between p-5 w-full",
            image && "lg:w-2/3"
          )}
        >
          <div>
            <div className="flex items-center gap-2 mb-1">
              <h3
                className="font-semibold text-base text-card-foreground truncate"
                title={title}
              >
                {title}
              </h3>
              {emojis.length > 0 && (
                <span className="flex gap-1 text-sm" title={emojis.join(" ")}>
                  {emojis.slice(0, 3).map((emoji, i) => (
                    <span key={i}>{emoji}</span>
                  ))}
                  {emojis.length > 3}
                </span>
              )}
            </div>
            <p className="text-sm text-muted-foreground leading-snug line-clamp-2">
              {preview}
            </p>
          </div>

          <div className="flex justify-between items-center mt-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Calendar className="w-4 h-4" />
              {date}
            </div>
            <div className="flex gap-2 items-center">
              <Button
                size="icon"
                variant="ghost"
                onClick={(e) => {
                  e.stopPropagation(); // Prevent card navigation
                  navigate(editUrl);
                }}
              >
                <Edit className="w-4 h-4" />
              </Button>

              <Button
                size="icon"
                variant="ghost"
                className="text-destructive hover:text-destructive"
                onClick={(e) => {
                  e.stopPropagation(); // Prevent card navigation
                  onDelete();
                }}
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </BaseCardContent>
      </div>
    </BaseCard>
  );
};
