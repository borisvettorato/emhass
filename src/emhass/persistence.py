"""Small persistent JSON blob storage under emhass_conf['data_path'].

Used for backend-owned state that needs to survive across optimization runs
but isn't part of the user-editable config.json (e.g. per-room comfort
schedules, legionella cycle completion timestamps).
"""

import os
import pathlib

import aiofiles
import orjson


async def save_json_blob(
    emhass_conf: dict,
    filename: str,
    data: dict,
    logger,
) -> bool:
    """Atomically write a JSON-serializable dict to data_path/filename.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :param filename: File name (not a path) to write under emhass_conf["data_path"]
    :param data: JSON-serializable dictionary to persist
    :param logger: Logger instance
    :return: True on success, False on failure
    """
    dest = pathlib.Path(emhass_conf["data_path"]) / filename
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        async with aiofiles.open(tmp, "wb") as f:
            await f.write(orjson.dumps(data))
        os.replace(tmp, dest)
        return True
    except Exception as e:
        logger.warning(f"Failed to save {filename} to {emhass_conf['data_path']}: {e}")
        return False


async def load_json_blob(
    emhass_conf: dict,
    filename: str,
    logger,
    default: dict | None = None,
) -> dict:
    """Load a JSON blob from data_path/filename, never raising.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :param filename: File name (not a path) to read under emhass_conf["data_path"]
    :param logger: Logger instance
    :param default: Value returned when the file is missing or unreadable
    :return: The parsed dictionary, or `default` (or {} if None) on any failure
    """
    if default is None:
        default = {}
    src = pathlib.Path(emhass_conf["data_path"]) / filename
    if not src.exists():
        return default
    try:
        async with aiofiles.open(src, "rb") as f:
            content = await f.read()
        return orjson.loads(content)
    except Exception as e:
        logger.warning(f"Failed to load {filename} from {emhass_conf['data_path']}: {e}")
        return default
