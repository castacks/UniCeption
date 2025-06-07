"""
Utilitary functions for visualizations
"""

from argparse import ArgumentParser, Namespace
from distutils.util import strtobool


def str2bool(v):
    return bool(strtobool(v))


def script_add_rerun_args(parser: ArgumentParser) -> None:
    """
    Add common Rerun script arguments to `parser`.

    Parameters
    ----------
    parser : ArgumentParser
        The parser to add arguments to.

    Returns
    -------
    None
    """
    parser.add_argument("--headless", type=str2bool, nargs="?", const=True, default=True, help="Don't show GUI")
    parser.add_argument(
        "--connect",
        dest="connect",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Connect to an external viewer",
    )
    parser.add_argument(
        "--serve",
        dest="serve",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Serve a web viewer (WARNING: experimental feature)",
    )
    parser.add_argument(
        "--addr",
        type=str,
        default="0.0.0.0:<rr-port>",
        help="Connect to this ip:port. Replace <rr-port> with the actual port number.",
    )
    parser.add_argument("--save", type=str, default=None, help="Save data to a .rrd file at this path")
    parser.add_argument(
        "-o",
        "--stdout",
        dest="stdout",
        action="store_true",
        help="Log data to standard output, to be piped into a Rerun Viewer",
    )


def init_rerun_args(
    headless=True, connect=True, serve=False, addr="0.0.0.0:<rr-port>", save=None, stdout=False
) -> Namespace:
    """
    Initialize common Rerun script arguments.

    Parameters
    ----------
    headless : bool, optional
        Don't show GUI, by default True
    connect : bool, optional
        Connect to an external viewer, by default True
    serve : bool, optional
        Serve a web viewer (WARNING: experimental feature), by default False
    addr : str, optional
        Connect to this ip:port, by default "0.0.0.0:<rr-port>". Replace <rr-port> with the actual port number.
    save : str, optional
        Save data to a .rrd file at this path, by default None
    stdout : bool, optional
        Log data to standard output, to be piped into a Rerun Viewer, by default False

    Returns
    -------
    Namespace
        The parsed arguments.
    """
    rerun_args = Namespace()
    rerun_args.headless = headless
    rerun_args.connect = connect
    rerun_args.serve = serve
    rerun_args.addr = addr
    rerun_args.save = save
    rerun_args.stdout = stdout

    return rerun_args
