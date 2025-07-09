import api from "./api";
import { GoalCreate, GoalUpdate, GoalResponse } from "../types/goal";

export const getAllGoals = async (
  skip = 0,
  limit = 100
): Promise<GoalResponse[]> => {
  const response = await api.get(`/goals?skip=${skip}&limit=${limit}`);
  return response.data;
};

export const getGoalById = async (goalId: string): Promise<GoalResponse> => {
  const response = await api.get(`/goals/${goalId}`);
  return response.data;
};

export const createGoal = async (data: GoalCreate): Promise<GoalResponse> => {
  const response = await api.post("/goals", data);
  return response.data;
};

export const updateGoal = async (
  goalId: string,
  data: GoalUpdate
): Promise<GoalResponse> => {
  const response = await api.put(`/goals/${goalId}`, data);
  return response.data;
};

export const deleteGoal = async (
  goalId: string
): Promise<{ detail: string }> => {
  const response = await api.delete(`/goals/${goalId}`);
  return response.data;
};

export const deleteAllGoals = async (): Promise<{ detail: string }> => {
  const response = await api.delete("/goals/all");
  return response.data;
};
