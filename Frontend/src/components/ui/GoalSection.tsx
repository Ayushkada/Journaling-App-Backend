import GoalCard from "@/components/ui/GoalsCard";
import { GoalBase } from "@/types/goal";

export const GoalSection = ({
  title,
  goals,
  onDelete,
}: {
  title: string;
  goals: GoalBase[];
  onDelete: (id: string) => void;
}) => (
  <section
    className="mb-10"
    aria-labelledby={`${title.toLowerCase().replace(/\s/g, "-")}-title`}
  >
    <h2
      id={`${title.toLowerCase().replace(/\s/g, "-")}-title`}
      className="text-2xl font-semibold text-foreground mb-4"
    >
      {title}
    </h2>
    <div className="grid md:grid-cols-2 gap-6">
      {goals.map((goal) => (
        <GoalCard
          key={goal.id}
          goal={goal}
          onDelete={() => onDelete(goal.id)}
          editUrl={`/goals/${goal.id}/edit`}
          viewUrl={`/goals/${goal.id}`}
        />
      ))}
    </div>
  </section>
);
