from colorama import Fore, Style


class ExtrasDependencyImportError(Exception):
    """Raisable on ImportErrors due to missing package extras."""

    def __init__(self, extras_type: str, nested_error: Exception):
        style: str = Style.BRIGHT + Fore.GREEN
        message: str = (
            f"\n\n📦 {nested_error}\n\n"
            + f"{Style.DIM}# Have you tried running the following?{Style.RESET_ALL}\n"
            + f"$ {style}pip install 'mother[{extras_type}]'{Style.RESET_ALL} or\n"
            + f"$ {style}uv add 'mother[{extras_type}]'{Style.RESET_ALL}"
        )
        super().__init__(message)


class ConfigurationError(Exception): ...
