import { useMutation } from '@tanstack/react-query';
import { guardrailApi } from '@/lib/api';
import { GuardrailRequest } from '@/lib/types';

export function useGuardrailApi() {
    return useMutation({
        mutationFn: async (data: GuardrailRequest) => {
            const response = await guardrailApi.validate(data);
            return response.data;
        },
    });
}
