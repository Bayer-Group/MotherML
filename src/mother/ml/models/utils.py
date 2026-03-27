def add_prefix_to_dict_keys(d: dict, prefix: str) -> dict:
    """
    Add a prefix to every key in a dictionary.

    Args:
        d: The original dictionary
        prefix: The prefix to add to each key

    Returns:
        A new dictionary with prefixed keys
    """
    return {prefix + key: value for key, value in d.items()}
