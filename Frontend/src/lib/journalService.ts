import {
  JournalEntryBase,
  JournalEntryCreate,
  JournalEntryUpdate,
} from "../types/journal";
import api from "./api";

export const getAllJournals = async (
  skip = 0,
  limit = 100
): Promise<JournalEntryBase[]> => {
  const response = await api.get(`/journals/all?skip=${skip}&limit=${limit}`);
  return response.data;
};

export const getJournalById = async (
  journalId: string
): Promise<JournalEntryBase> => {
  const response = await api.get(`/journals/${journalId}`);
  return response.data;
};

export const createJournal = async (
  data: JournalEntryCreate
): Promise<JournalEntryBase> => {
  const response = await api.post("/journals", data);
  return response.data;
};

export const updateJournal = async (
  journalId: string,
  data: JournalEntryUpdate
): Promise<JournalEntryBase> => {
  const response = await api.put(`/journals/${journalId}`, data);
  return response.data;
};

export const deleteJournal = async (
  journalId: string
): Promise<{ detail: string }> => {
  const response = await api.delete(`/journals/${journalId}`);
  return response.data;
};

export const deleteAllJournals = async (): Promise<{ detail: string }> => {
  const response = await api.delete("/journals/all");
  return response.data;
};
