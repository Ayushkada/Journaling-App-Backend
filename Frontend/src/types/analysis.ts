export interface JournalAnalysisBase {
  id: string;
  journal_id: string;
  user_id: string;
  analysis_date: string;
  readability: number;
  sentiment_score: number;
  self_talk_tone: string;
  energy_score: number;
  keywords: Record<string, number>;
  text_mood: Record<string, number>;
  emoji_mood: Record<string, number>;
  image_mood: Record<string, number>;
  combined_mood: Record<string, number>;
  goal_mentions: string[];
  topics: { [key: string]: string }[]; // or refine if possible
  text_vector: string;
  text_embedding: number[];
  extracted_actions: string;
  date: string;
  model: string;
}

export interface ConnectedAnalysisBase {
  id: string;
  user_id: string;
  created_at: string;
  mood_trends: Record<string, Record<string, number>>;
  energy_trends: Record<string, number>;
  average_sentiment: number;
  goal_emotion_map: Record<string, Record<string, number>>;
  goal_progress: Record<string, Record<string, any>>;
  goal_matches: Record<string, string[]>;
  keyword_emotion_map: Record<string, Record<string, number>>;
  keyword_energy_map: Record<string, number>;
  journal_weights: Record<string, number>;
  model: string;
}

export interface FeedbackBase {
  id: string;
  user_id: string;
  connected_analysis_id: string;
  tone: string;
  feedback: string;
  reflective_question: string;
  motivation: string;
  created_at: string;
}
