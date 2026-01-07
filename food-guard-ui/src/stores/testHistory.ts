import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { TestHistory } from '@/lib/types';

interface TestHistoryState {
    history: TestHistory[];
    addTest: (test: TestHistory) => void;
    clearHistory: () => void;
}

export const useTestHistoryStore = create<TestHistoryState>()(
    persist(
        (set) => ({
            history: [],
            addTest: (test) => set((state) => ({
                history: [test, ...state.history].slice(0, 1000)
            })),
            clearHistory: () => set({ history: [] }),
        }),
        {
            name: 'food-guard-history',
        }
    )
);
