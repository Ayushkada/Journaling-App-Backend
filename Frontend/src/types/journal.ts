export interface JournalEntryBase {
  id: string; // UUID
  user_id: string; // UUID
  title?: string;
  content: string;
  updated_date?: string; // ISO string (datetime)
  date: string; // ISO string (datetime)
  emojis?: string[];
  images?: string[];
  analyze_images: boolean;
  source: string;
  analysis_status?: string; // "pending" or similar
}

export interface JournalEntryCreate {
  title?: string;
  content: string;
  date: string; // ISO string (use `new Date().toISOString()` in frontend)
  emojis?: string[];
  images?: string[];
  analyze_images?: boolean;
  source: string;
}

export interface JournalEntryUpdate {
  title?: string;
  content?: string;
  updated_date: string; // ISO string (datetime)
  date?: string;
  emojis?: string[];
  images?: string[];
  analyze_images?: boolean;
  source?: string;
  analysis_status?: string;
}
