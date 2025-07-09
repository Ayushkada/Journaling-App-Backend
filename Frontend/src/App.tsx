import React from "react";
import { BrowserRouter, useRoutes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider } from "@/provider";
import { routes } from "@/routes";
import { Toaster } from "sonner";
import { TooltipProvider } from "@radix-ui/react-tooltip";

const queryClient = new QueryClient();

function AppRoutes() {
  return useRoutes(routes);
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <BrowserRouter>
          <AuthProvider>
            <AppRoutes />
          </AuthProvider>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
}
