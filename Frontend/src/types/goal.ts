export interface GoalBase {
  id: string;
  user_id: string;
  parent_goal_id?: string | null;
  content: string;
  ai_generated: boolean;
  category?: string[] | null;
  completed_at?: string | null;
  created_at: string;
  updated_at?: string | null;
  emotion_trend?: number[] | null;
  first_mentioned_at?: string | null;
  last_mentioned_at?: string | null;
  notes?: string | null;
  progress_score: number;
  related_entry_ids?: string[] | null;
  time_limit?: string | null;
  verified: boolean;
}

export interface GoalCreate {
  content: string;
  ai_generated: boolean;
  category?: string[] | null;
  completed_at?: string;
  created_at: string;
  emotion_trend?: number[];
  first_mentioned_at?: string;
  last_mentioned_at?: string;
  notes?: string;
  progress_score: number;
  related_entry_ids?: string[];
  time_limit?: string;
  verified?: boolean;
  parent_goal_id?: string;
}

export interface GoalUpdate {
  content?: string;
  ai_generated?: boolean;
  category?: string[] | null;
  updated_at: string;
  completed_at?: string;
  emotion_trend?: number[];
  first_mentioned_at?: string;
  last_mentioned_at?: string;
  notes?: string;
  progress_score?: number;
  related_entry_ids?: string[];
  time_limit?: string;
  verified?: boolean;
  parent_goal_id?: string;
}

export type GoalResponse = GoalBase;
