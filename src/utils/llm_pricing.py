# from https://openai.com/api/pricing/
openai_api_rates_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-2024-05-13": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-11-20": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gpt-4o-mini-2024-07-18": {"input": 0.150, "output": 0.600},
    # ^^^ [Sept2025] gpt4o prices might be outdated ^^^
    "gpt-5": {"input": 1.50, "output": 10.00},
    "gpt-5-2025-08-07": {"input": 1.50, "output": 10.00},
}


def get_price(completion):
    completion = to_dict(completion)
    try:
        model = completion["model"]
        usage = completion["usage"]
    except KeyError as e:
        print("Failed to get model and usage from completion object.")
        print(f"KeyError: {e}")
        print("Completion object:", completion)
        print("Returning 0 as price.")
        return 0
    input_price = openai_api_rates_1M[model]["input"] * usage["prompt_tokens"] / 1_000_000
    output_price = openai_api_rates_1M[model]["output"] * usage["completion_tokens"] / 1_000_000
    return input_price + output_price


def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return to_dict(vars(obj))
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_dict(v) for v in obj)
    return obj
