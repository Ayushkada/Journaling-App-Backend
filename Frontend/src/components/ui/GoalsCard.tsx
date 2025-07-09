import { GoalBase } from "@/types/goal";
import { Edit, Trash2, Calendar } from "lucide-react";
import { getCategoryColor } from "@/utils/helpers";
import { Button } from "./Button";
import { useNavigate } from "react-router-dom";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";

interface GoalCardProps {
  goal: GoalBase;
  editUrl: string;
  viewUrl: string;
  onDelete: () => void;
}

const GoalCard: React.FC<GoalCardProps> = ({
  goal,
  onDelete,
  editUrl,
  viewUrl,
}) => {
  const navigate = useNavigate();

  const handleCardClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (target.closest("button")) return;
    navigate(viewUrl);
  };

  return (
    <div
      onClick={handleCardClick}
      role="button"
      tabIndex={0}
      className="bg-card rounded-lg p-6 border hover:scale-[1.02] shadow-md transition-all flex flex-col justify-between h-full"
    >
      <div className="flex flex-col sm:flex-row justify-between gap-4 h-full overflow-hidden">
        {/* Left side */}
        <div className="flex flex-col justify-between flex-1 overflow-hidden">
          <div className="overflow-hidden no-scrollbar">
            <div
              className="flex gap-2 mb-2 overflow-x-auto whitespace-nowrap max-w-full"
              style={{
                scrollbarWidth: "none",
                msOverflowStyle: "none",
                WebkitOverflowScrolling: "touch",
                minHeight: "1.5rem",
              }}
            >
              {goal.category?.length > 0 &&
                goal.category.map((cat) => (
                  <span
                    key={cat}
                    className={`text-xs font-medium px-2 py-1 rounded-full inline-block ${getCategoryColor(
                      cat
                    )}`}
                  >
                    {cat}
                  </span>
                ))}

              <style>
                {`
                  div::-webkit-scrollbar {
                    display: none;
                  }
                `}
              </style>
            </div>

            <div className="relative group">
              <h3 className="text-base font-semibold text-card-foreground mb-1 line-clamp-2">
                {goal.content}
              </h3>

              {goal.notes && !goal.completed_at && (
                <p className="text-muted-foreground text-sm line-clamp-1">
                  {goal.notes}
                </p>
              )}
            </div>
          </div>

          <div className="mt-4 text-sm text-muted-foreground space-y-1">
            {goal.completed_at ? (
              <div className="flex items-center">
                <Calendar className="w-4 h-4 mr-1" />
                Completed on: {new Date(goal.completed_at).toLocaleDateString()}
              </div>
            ) : (
              <>
                {goal.time_limit && (
                  <div className="flex items-center">
                    <Calendar className="w-4 h-4 mr-1" />
                    Due: {new Date(goal.time_limit).toLocaleDateString()}
                  </div>
                )}
                <div>
                  Last Updated:{" "}
                  {goal.updated_at
                    ? new Date(goal.updated_at).toLocaleDateString()
                    : new Date(goal.created_at).toLocaleDateString()}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Right side */}
        <div className="flex flex-col items-end justify-between gap-3">
          <div className="flex">
            <Button
              size="icon"
              variant="ghost"
              aria-label="Edit goal"
              onClick={(e) => {
                e.stopPropagation();
                navigate(editUrl);
              }}
            >
              <Edit className="w-4 h-4" />
            </Button>

            {!goal.completed_at && (
              <Button
                size="icon"
                variant="ghost"
                className="text-destructive hover:text-destructive"
                aria-label="Delete goal"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete();
                }}
              >
                <Trash2 className="w-4 h-4" />
              </Button>
            )}
          </div>

          <div className="w-20 h-20 flex flex-col items-center justify-between">
            <CircularProgressbar
              value={goal.progress_score}
              text={`${goal.progress_score}%`}
              styles={buildStyles({
                pathColor: "hsl(var(--primary))",
                textColor: "hsl(var(--foreground))",
                trailColor: "hsl(var(--muted))",
                textSize: "20px",
              })}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default GoalCard;
