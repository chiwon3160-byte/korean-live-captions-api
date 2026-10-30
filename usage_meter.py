"""In-memory API usage estimates; never accepts keys, prompts or responses.

USD per million tokens, standard rates verified 2026-10-30:
https://developers.openai.com/api/docs/models/gpt-6-luna
https://developers.openai.com/api/docs/models/gpt-6.1-terra
https://developers.openai.com/api/docs/models/gpt-6.1-sol
Cached input discounts, taxes and final provider billing are not included.
"""
from decimal import Decimal
import threading

RATES = {
    ('openai', 'gpt-6-luna'): (Decimal('0.1'), Decimal('0.5')),
    ('openai', 'gpt-6.1-terra'): (Decimal('1'), Decimal('5')),
    ('openai', 'gpt-6.1-sol'): (Decimal('2'), Decimal('10')),
}
LABELS = {'gpt-6-luna': 'Luna 6', 'gpt-6.1-terra': 'Terra 6.1', 'gpt-6.1-sol': 'SOL 6.1'}


class UsageMeter:
    def __init__(self):
        self.models = {}
        self._lock = threading.Lock()

    def record(self, provider, model, usage, success):
        if provider == 'local':
            return
        with self._lock:
            row = self.models.setdefault((provider, model), {
                'calls': 0, 'successes': 0, 'input_tokens': 0, 'output_tokens': 0,
                'unknown_calls': 0, 'estimated_usd': Decimal('0')})
            row['calls'] += 1
            row['successes'] += int(bool(success))
            valid = isinstance(usage, dict) and all(
                type(usage.get(k)) is int and usage[k] >= 0
                for k in ('prompt_tokens', 'generated_tokens'))
            if valid:
                row['input_tokens'] += usage['prompt_tokens']
                row['output_tokens'] += usage['generated_tokens']
            rates = RATES.get((provider, model))
            if not valid or rates is None:
                row['unknown_calls'] += 1
            else:
                # Reasoning tokens already belong to output_tokens; never add again.
                row['estimated_usd'] += (usage['prompt_tokens'] * rates[0]
                                         + usage['generated_tokens'] * rates[1]) / Decimal(1000000)

    def summary(self):
        with self._lock:
            rows = [(key, dict(value)) for key, value in self.models.items()]
        total = sum((row['estimated_usd'] for _, row in rows), Decimal('0'))
        unknown = sum(row['unknown_calls'] for _, row in rows)
        lines = [f'이번 실행 API 예상 비용: ${total:.6f}'
                 + (f' + 비용 미확인 {unknown}건' if unknown else '')]
        for (provider, model), row in rows:
            label = LABELS.get(model, model)
            cost = (f'${row["estimated_usd"]:.6f}' if (provider, model) in RATES
                    else '단가 미등록')
            lines.append(f'{label}: {cost} · 요청 {row["calls"]}건 / 번역 성공 {row["successes"]}건'
                         f' · 입력 {row["input_tokens"]:,} / 출력 {row["output_tokens"]:,} 토큰'
                         + (f' · 비용 미확인 {row["unknown_calls"]}건' if row['unknown_calls'] else ''))
        lines.append('앱 실행 중 누적 · 응답 사용량 기준 · 할인·세금 제외 / 최종 청구액 아님')
        return '\n'.join(lines)
