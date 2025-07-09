import api from "./api";
import {
  JournalAnalysisBase,
  ConnectedAnalysisBase,
  FeedbackBase,
} from "../types/analysis";

export const runFullAnalysis = async (
  model: string = "local-small",
  feedback_tone: string = "calm",
  prompt_tone: string = "friendly"
): Promise<any> => {
  const response = await api.post(
    `/analysis/run?model=${model}&feedback_tone=${feedback_tone}&prompt_tone=${prompt_tone}`
  );
  return response.data;
};

export const analyzeRecentJournals = async (
  model: string = "local-small"
): Promise<any> => {
  const response = await api.post(`/analysis/journals?model=${model}`);
  return response.data;
};

export const getAllJournalAnalyses = async (
  skip = 0,
  limit = 100
): Promise<JournalAnalysisBase[]> => {
  const response = await api.get(
    `/analysis/journals/all?skip=${skip}&limit=${limit}`
  );
  return response.data;
};

export const getJournalAnalysisById = async (
  journalId: string
): Promise<JournalAnalysisBase> => {
  const response = await api.get(`/analysis/journals/${journalId}`);
  return response.data;
};

export const deleteJournalAnalysis = async (
  journalId: string
): Promise<{ detail: string }> => {
  const response = await api.delete(`/analysis/journals/${journalId}`);
  return response.data;
};

export const generateConnectedAnalysis = async (
  model: string = "local-small"
): Promise<any> => {
  const response = await api.post(`/analysis/connected?model=${model}`);
  return response.data;
};

export const getConnectedAnalysis =
  async (): Promise<ConnectedAnalysisBase> => {
    const response = await api.get("/analysis/connected");
    return response.data;
  };

export const deleteConnectedAnalysis = async (): Promise<{
  detail: string;
}> => {
  const response = await api.delete("/analysis/connected");
  return response.data;
};

export const generateFeedback = async (
  tone: string = "calm",
  model: string = "local-small"
): Promise<any> => {
  const response = await api.post(
    `/analysis/feedback?model=${model}&tone=${tone}`
  );
  return response.data;
};

export const getFeedback = async (): Promise<FeedbackBase> => {
  const response = await api.get("/analysis/feedback");
  return response.data;
};

export const generatePrompts = async (
  tone: string = "friendly",
  model: string = "local-small"
): Promise<any> => {
  const response = await api.post(
    `/analysis/prompts?model=${model}&tone=${tone}`
  );
  return response.data;
};

export const generateGoalsFromAI = async (
  model: string = "local-small"
): Promise<any> => {
  const response = await api.post(`/analysis/goals?model=${model}`);
  return response.data;
};
