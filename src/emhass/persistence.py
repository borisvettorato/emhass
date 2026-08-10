"""Small persistent blob storage under emhass_conf['data_path'].

Used for backend-owned state that needs to survive across optimization runs
but isn't part of the user-editable config.json (e.g. per-room comfort
schedules, legionella cycle completion timestamps, fitted model objects).
"""

import os
import pathlib
import pickle

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


async def save_pickle_blob(
    emhass_conf: dict,
    filename: str,
    obj: object,
    logger,
) -> bool:
    """Atomically pickle an arbitrary object to data_path/filename.

    Same atomic tmp+rename shape as save_json_blob, for objects that aren't
    JSON-serializable (e.g. fitted sklearn Pipeline objects).

    :param emhass_conf: Dictionary containing the needed emhass paths
    :param filename: File name (not a path) to write under emhass_conf["data_path"]
    :param obj: Picklable object to persist
    :param logger: Logger instance
    :return: True on success, False on failure
    """
    dest = pathlib.Path(emhass_conf["data_path"]) / filename
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        async with aiofiles.open(tmp, "wb") as f:
            await f.write(pickle.dumps(obj))
        os.replace(tmp, dest)
        return True
    except Exception as e:
        logger.warning(f"Failed to save {filename} to {emhass_conf['data_path']}: {e}")
        return False


async def load_pickle_blob(
    emhass_conf: dict,
    filename: str,
    logger,
    default: object | None = None,
) -> object | None:
    """Load a pickled object from data_path/filename, never raising.

    :param emhass_conf: Dictionary containing the needed emhass paths
    :param filename: File name (not a path) to read under emhass_conf["data_path"]
    :param logger: Logger instance
    :param default: Value returned when the file is missing or unreadable
    :return: The unpickled object, or `default` on any failure
    """
    src = pathlib.Path(emhass_conf["data_path"]) / filename
    if not src.exists():
        return default
    try:
        async with aiofiles.open(src, "rb") as f:
            content = await f.read()
        return pickle.loads(content)
    except Exception as e:
        logger.warning(f"Failed to load {filename} from {emhass_conf['data_path']}: {e}")
        return default
