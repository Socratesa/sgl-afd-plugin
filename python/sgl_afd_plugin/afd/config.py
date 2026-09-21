from dataclasses import dataclass

from sgl_afd_plugin.afd.utils.config_utils import A, Arg, plugin_config


@dataclass
class AFDConfig:
    """Sglang AFD plugin config.

    A field is passed as a CLI flag named by the prefix (`afd`):

        sglang serve --model-path <model> --afd-role attn

    Use get_config() reads it back in any process; Each value is passed to its
    subprocesses as AFD_<FIELD> (e.g. AFD_ROLE).

    Declaration forms:

        name: A[T, "help text"] = default
        name: A[T, Arg(help="...", choices=[...], aliases=["--x"])] = default

    Arg knobs: help, choices, aliases, cli_name, nargs, type_parser, required,
        action, action_kwargs, const, no_cli.
    Types: bool / int / float / str / Literal[...] / list | tuple | set[scalar] /
        dict (given as JSON).
    """

    role: A[
        str,
        Arg(
            help="Role in Attention-FFN Disaggregation Deployment",
            choices=["attn", "ffn"],
        )
    ]


AFD_CONFIG = plugin_config(AFDConfig, prefix="afd")


def get_config() -> AFDConfig:
    """Get sglang Attention-FFN Disaggregation (AFD) plugin config."""
    return AFD_CONFIG.read()
