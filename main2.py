"""main script for UniTSyncer backend"""
from tqdm import tqdm
from unitsyncer.sync import Synchronizer, LSPSynchronizer
from unitsyncer.rust_syncer import RustSynchronizer
from unitsyncer.sansio_lsp_syncer import SansioLSPSynchronizer
from pylspclient.lsp_structs import LANGUAGE_IDENTIFIER, Location, Position, Range
from returns.maybe import Maybe, Nothing, Some
from returns.result import Result, Success, Failure
from unitsyncer.util import parallel_starmap as starmap, path2uri, convert_to_seconds
from unitsyncer.common import CORES
from unitsyncer.extract_def import get_def_header
import math
from unitsyncer.source_code import get_function_code
import json
import jsonlines
import os
from pathos.multiprocessing import ProcessPool
import logging
import fire
from itertools import groupby
import pathlib


def id2path(func_id: str) -> str:
    return func_id.split("::")[0]


def wrap_repo(repo_id: str) -> str:
    """Convert repository ID to directory name format"""
    return repo_id.replace("/", "-")


def java_workdir_dict(objs: list[dict]) -> dict[str, list[dict]]:
    """split a list of test ids into a dict of workdir to file path
    this solves the LSP TimeoutError for JAVA with too much subdirectories

    Args:
        objs (list[dict]): [focal_ids parsed into dict]

    Returns:
        dict[str, list[dict]]: {workdir: [corresponding focal objects, ...], ...}
    """
    workdir_dict: dict[str, list[dict]] = {}
    for obj in objs:
        test_id = obj["test_id"]
        file_path = id2path(test_id)
        workdir = file_path.split("/test")[0]
        if workdir not in workdir_dict:
            workdir_dict[workdir] = []
        workdir_dict[workdir].append(obj)
    return workdir_dict


def focal2result(syncer: Synchronizer, repos_root, obj):
    logging.debug(f"Processing test_id: {obj['test_id']}")
    p = id2path(obj["test_id"])
    file_path = os.path.join(repos_root, p)
    print(f"Looking for file: {file_path}")
    print(f"File exists: {os.path.exists(file_path)}")
    src_lineno, src_col_offset = obj["focal_loc"]
    test_lineno, test_col_offset = obj["test_loc"]
    print(f"Looking for focal_id: {obj['focal_id']} at location: {obj['focal_loc']}")

    langID = syncer.langID

    # only python ast is 1-indexed, tree-sitter and LSP are 0-indexed
    match langID:
        case LANGUAGE_IDENTIFIER.PYTHON:
            src_lineno -= 1
            test_lineno -= 1

    # since the test's delc node is already capture by frontend, it can store the test code
    if "test" in obj.keys():
        test = obj["test"]
    else:
        fake_loc = Location(
            path2uri(file_path),
            Range(
                Position(test_lineno, test_col_offset),
                Position(test_lineno, test_col_offset + 1),
            ),
        )
        test, _, _ = get_function_code(fake_loc, syncer.langID).unwrap()

    result = {
        "test_id": obj["test_id"],
        "test": test,
    }

    print(f"Attempting to get source for focal_id: {obj['focal_id']}")
    match syncer.get_source_of_call(
        obj["focal_id"],
        file_path,
        src_lineno,
        src_col_offset,
    ):
        case Success((code, docstring, code_id)):
            print(f"Successfully found definition:")  # 添加日志
            print(f"code: {code[:50]}...")  # 添加日志，只打印前100个字符
            print(f"docstring: {docstring:50}")
            print(f"code_id: {code_id}")
            result["code_id"] = (
                obj["focal_id"]
                if code_id is None
                else code_id.removeprefix(repos_root + "/")
            )
            result["code"] = code
            result["docstring"] = docstring
            result["test_header"] = get_def_header(test, langID)
        case Failure(e):
            print(f"Failed to get source: {e}")  # 添加日志
            logging.debug(e)
            result["error"] = e

    print(f"Final result: {result}")  # 添加日志，查看最终结果
    return result


def process_one_focal_file(
        focal_file: str,
        repos_root: str,
        language: str,
        success_file: str,
        failure_file: str,
        skip_processed: bool = True,
) -> tuple[int, int]:
    with open(focal_file) as f:
        objs = [json.loads(line) for line in f.readlines()]

    if len(objs) == 0:
        return 0, 0

    n_focal = len(objs)
    match language:
        case LANGUAGE_IDENTIFIER.JAVA:
            wd = java_workdir_dict(objs)
        case _:
            first_test_id = objs[0]["test_id"]
            workdir = "/".join(id2path(first_test_id).split("/")[:2])
            wd = {
                workdir: objs,
            }

    success_results = []
    failure_results = []

    # check if this file is already processed
    if skip_processed and os.path.exists(success_file) and os.path.exists(failure_file):
        with open(success_file, "rb") as f:
            n_succ = sum(1 for _ in f)
        with open(failure_file, "rb") as f:
            n_fail = sum(1 for _ in f)

        if language == LANGUAGE_IDENTIFIER.JAVA:
            return n_focal, n_succ
        if n_succ + n_fail >= n_focal:
            return n_focal, n_succ

    pathlib.Path(success_file).touch()
    pathlib.Path(failure_file).touch()

    logging.debug(f"number of workdir_dict: {len(wd.keys())}")
    repos_root = os.path.abspath(repos_root)
    for workdir, workdir_objs in wd.items():
        print(f"\nProcessing workdir: {workdir}")
        print(f"Full workdir path: {os.path.join(repos_root, workdir)}")
        print(f"Contents of workdir:")
        try:
            print(os.listdir(os.path.join(repos_root, workdir)))
        except Exception as e:
            print(f"Error listing directory: {e}")
        succ = []
        fail = []
        full_workdir = os.path.join(repos_root, workdir)
        logging.debug(f"workdir: {full_workdir}")
        syncer: Synchronizer

        match language:
            case LANGUAGE_IDENTIFIER.RUST:
                syncer = RustSynchronizer(full_workdir, language)
            case LANGUAGE_IDENTIFIER.GO:
                syncer = SansioLSPSynchronizer(full_workdir, language)
            case _:
                syncer = LSPSynchronizer(full_workdir, language)

        try:
            syncer.initialize(timeout=120)

            for obj in workdir_objs:
                result = focal2result(syncer, repos_root, obj)
                if "error" in result:
                    fail.append(result)
                else:
                    succ.append(result)

            syncer.stop()
        except Exception as e:
            logging.debug(e)
            syncer.stop()
            continue

        # append to source file in loop to avoid losing data
        with jsonlines.open(success_file, "a") as f:
            f.write_all(succ)
            print(f"Wrote {len(succ)} successful results")
        with jsonlines.open(failure_file, "a") as f:
            f.write_all(fail)
            print(f"Wrote {len(fail)} failed results")

        success_results.extend(succ)
        failure_results.extend(fail)

    return n_focal, len(success_results)


def get_repo_focal_files(focal_path: str, repo_id_list: list[str]) -> list[str]:
    """Get all focal files for specified repositories"""
    focal_files = []
    for repo_id in repo_id_list:
        repo_focal_file = os.path.join(focal_path, wrap_repo(repo_id) + ".jsonl")
        if os.path.exists(repo_focal_file):
            focal_files.append(repo_focal_file)
    return focal_files


def main(
        repo_id: str = "mistifyio/go-zfs",
        repos_root: str = "data/repos/",
        focal_path: str = "data/focal/",
        language: str = "python",
        jobs: int = CORES,
        debug: bool = False,
        timeout: str = "30m",
):
    """Main function to process repositories and extract source code

    Args:
        repo_id (str): Path to file containing repository IDs or single repository ID
        repos_root (str): Root directory containing all repositories
        focal_path (str): Directory containing focal files
        language (str): Programming language
        jobs (int): Number of parallel jobs
        debug (bool): Enable debug logging
        timeout (str): Timeout duration
    """
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)

    # Load repository IDs
    try:
        repo_id_list = [l.strip() for l in open(repo_id, "r").readlines()]
        # 从 repo_id 文件路径中提取基础文件名
        base_name = os.path.splitext(os.path.basename(repo_id))[0]
    except FileNotFoundError:
        repo_id_list = [repo_id]
        base_name = "repo_id"

    logging.info(f"Processing repositories: {len(repo_id_list)}")

    # Get focal files for specified repositories
    focal_files = get_repo_focal_files(focal_path, repo_id_list)
    if not focal_files:
        logging.error(f"No focal files found for the specified repositories in {focal_path}")
        return

    logging.info(f"Found {len(focal_files)} focal files to process")

    # 创建输出目录
    os.makedirs("data/source", exist_ok=True)

    # 设置输出文件路径
    success_file = os.path.join("data/source", f"{base_name}_success.jsonl")
    failure_file = os.path.join("data/source", f"{base_name}_failure.jsonl")

    # Process focal files with parallel jobs
    with ProcessPool(math.ceil(jobs / 2)) as pool:
        rnt = list(
            tqdm(
                pool.imap(
                    lambda f: process_one_focal_file(
                        f,
                        repos_root=repos_root,
                        language=language,
                        success_file=success_file,
                        failure_file=failure_file
                    ),
                    focal_files,
                ),
                total=len(focal_files),
            )
        )

    nfocal, ncode = zip(*rnt)
    logging.info(
        f"Processed {sum(ncode)} have source code in {sum(nfocal)} focal functions"
    )


if __name__ == "__main__":
    fire.Fire(main)