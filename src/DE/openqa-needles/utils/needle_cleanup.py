#! /usr/bin/env python3

"""
Needle cleanup

This script provides the ability to remove unused and/or expired needles
from a branch of the needles repository. It can also identify any missing
needles used by the provided OpenQA test repository, and needles that use
badly formatted datetimes, i.e not <needle_basename>-<datetime>.

Note: Any untracked needle files will not be deleted.

Required Python Modules:

- pip install GitPython (https://pypi.org/project/GitPython/)
- pip install python-gitlab (https://pypi.org/project/python-gitlab/)

Usage:

    python needles_cleanup.py <path-to-tests-repo> <path-to-needles-repo>

Optional arguments:

--summary                 : Displays a summary of the needle states,
                            and attempts no needle deletion
--expiry_date <YYYYMMDD>  : A datetime before which needles are considered expired
--branch <branch-name>    : The branch name of the needle repository to work on.
                            Currently defaults to 'master'
--create-merge-request    : Use if you want cleanup to create an MR, in addition
                            to the creation of a commit
--token <access-token>    : Gitlab access token, required to generate MR

When needles are deleted, a new branch of the needles repository is created and
a commit generated for the deleted json/png pairs. The script can also generate
an MR for this commit if the --create-merge-request and --token flags arguments
are provided.

Example:

    python needles_cleanup.py <path-to-tests-repo> <path-to-needles-repo> --expiry_date 20231201

A console menu will appear, and provided there are unused and expired needles,
would look like:

    What would you like to do?
    [u] Delete unused needles
    [e] Delete expired needles
    [q] Quit
    :

If a deletion choice is chosen, then a further menu will appear, offering the
choice of bulk deletion or individually selecting needles for deletion, like so:

    Deleting expired needles
    (b)ulk delete needles, (s)elect individually, or (r)eturn to menu:

If individual selection is chosen:

    Delete: app_baobab_home-20230423 (y/N):
    Delete: app_baobab_home-20230712 (y/N):
    Delete: app_baobab_home-20230819 (y/N):
    Delete: app_gnome_calculator_home-20230423 (y/N):
    etc ...

After selection, a msg containing the newly created branch with the delete
commit in it is displayed:

    Created branch: needles-cleanup/delete-expired-needles-1711027736, with a commit for the deletion of the needles

Note: Each branch is uniquely identified with an epoch timestamp.

"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple

import git
import gitlab

DATETIME_FORMAT = "YYYYMMDD"
GITLAB_URL = "https://gitlab.gnome.org"
NEEDLE_REPO_BRANCH = "master"
PROJECT_ID = "16754"


def argument_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Needle cleanup tool")
    parser.add_argument(
        "tests",
        type=str,
        help="The path to the OpenQA tests",
    )
    parser.add_argument(
        "needles",
        type=str,
        help="The path to the OpenQA needles",
    )
    parser.add_argument(
        "--expiry_date",
        default=None,
        type=str,
        help=f"Latest date to keep a needle from. Using format: {DATETIME_FORMAT}.",
    )
    parser.add_argument(
        "--branch",
        default=NEEDLE_REPO_BRANCH,
        type=str,
        help="Set the branch of the needle repository to work on.",
    )
    parser.add_argument(
        "--token",
        default="",
        type=str,
        help="Needle repository access token",
    )
    parser.add_argument(
        "--summary",
        default=False,
        action="store_true",
        help="Only give a summary of the needle states",
    )
    parser.add_argument(
        "--create-merge-request",
        default=False,
        action="store_true",
        help="Create a merge request after deletion of needles and making commit",
    )
    return parser


def str_to_datetime(date_string: str) -> datetime:
    """Return date string (YYYYMMDD) in datetime format."""
    return datetime.strptime(date_string, "%Y%m%d")


def get_test_needles(path: str, test_calls: List[str]) -> Set[str]:
    """Searches tests folder for specific calls and makes list
    of associated needles used by the calls.
    """
    os.chdir(path)
    files = os.listdir()
    needle_calls = []
    for file_name in files:
        abs_path = os.path.abspath(file_name)
        if os.path.isfile(abs_path):
            # Search tests for needle strings
            with open(file_name, "r", encoding="utf-8") as f:
                for line in f:
                    for text in test_calls:
                        if text in line:
                            if line.split("'")[0].strip()[:-1] == text:
                                needle_calls.append(line.split("'")[1])
    return set(needle_calls)


def get_needle_list(path: str) -> Tuple[List[str], List[str]]:
    """Returns a list of all stored needle names.
    A correct needle is assumed to have both a '.png' and '.json' file.
    """
    os.chdir(path)
    needle_list = []
    missing_json = []
    for file_name in glob.glob(path + "/*.png"):
        if Path(file_name[:-4] + ".json").is_file():
            needle_list.append(file_name.split("/")[-1][:-4])
        else:
            missing_json.append(file_name.split("/")[-1][:-4])

    return (needle_list, missing_json)


def check_missing_needles(needle_calls: List[str], needles: List[str]) -> List[str]:
    """Returns a list of missing needles."""
    missing = []
    for call in needle_calls:
        contains = False
        for needle in needles:
            if needle.startswith(call):
                contains = True
        if not contains:
            missing.append(call)
    return missing


def find_needle_states(
    needle_calls: List[str], needles_list: List[str]
) -> Tuple[Set[str], Set[str], List[str]]:
    """Returns a tuple of used needles, unused needles, and needles with badly formatted names."""
    badly_formatted_needle_names = []
    unused_needles = set()
    used_needles = set()
    for needle in needles_list:
        needle_base_name = needle.split("-")[0]
        if needle_base_name not in needle_calls:
            # Catch badly formatted needle names. Base names shouldn't end with a datetime
            if not needle_base_name[-len(DATETIME_FORMAT) :].isdigit():
                unused_needles.add(needle_base_name)
            else:
                badly_formatted_needle_names.append(needle_base_name)
            # Catch datetime versions of unused needles
            if (
                needle[-len(DATETIME_FORMAT) :].isdigit()
                and needle_base_name not in badly_formatted_needle_names
            ):
                unused_needles.add(needle)
        else:
            used_needles.add(needle)
    return (used_needles, unused_needles, badly_formatted_needle_names)


def find_expired_needles(
    used_needles: List[str], needle_calls: List[str], expiry_date: str
) -> List[str]:
    """Create a list of needles with a datetime older than that provided by expiry date.
    Determine if they should be removed and the needle repository updated.
    """
    expired_needles = []
    for call in needle_calls:
        needle_group = []
        expired_needle_group = []
        for needle in used_needles:
            if (
                call in needle
                and needle[-len(DATETIME_FORMAT) :].isdigit()
                and call == needle[: -(len(DATETIME_FORMAT) + 1)]
            ):
                needle_group.append(needle)
        # We probably only want to expire needles with more than two instances.
        # Namely the base needle name and another from a different datetime
        if len(needle_group) > 2:
            for needle in needle_group:
                if str_to_datetime(needle[-len(DATETIME_FORMAT) :]) < str_to_datetime(
                    expiry_date
                ):
                    expired_needle_group.append(needle)
        if expired_needle_group:
            for expired_needle in sorted(expired_needle_group):
                expired_needles.append(expired_needle)
    return expired_needles


def print_needle_summary(
    missing: List[str],
    badly_formatted: List[str],
    unused_needles: List[str],
    expired_needles: List[str],
    missing_json: List[str],
    expiry_date: str,
) -> None:
    """Prints a summary of needle states."""
    print("\033[1mNeedle Status Summary:\033[0m")
    if missing:
        print("\nThe following needle calls have no associated needles:")
        for miss in missing:
            print(miss)
    if badly_formatted:
        print(
            f"\nThere are \033[1m{len(badly_formatted)}\033[0m badly"
            " formatted needle names (Consider using the format '<needle_basename>-<datetime>'):"
        )
        for bad in badly_formatted:
            print(bad)
    if missing_json:
        print("\nThe following needle images do not have a json file:")
        for incomplete_needle in missing_json:
            print(incomplete_needle)
    if unused_needles:
        print(f"\nThere are \033[1m{len(unused_needles)}\033[0m unused needles:")
        for unused in sorted(unused_needles):
            print(unused)
    else:
        print("\nNo unused needles were found.")

    if expired_needles:
        print(
            f"\nThe following needles are older than the provided "
            f"expiry date ({expiry_date}):"
        )
        for expired_needle in expired_needles:
            print(expired_needle)
    elif expiry_date and not expired_needles:
        print("\nNo expired needles were found.")


def select_for_deletion(needle_list: List[str]) -> List[str]:
    """Decide to bulk delete needles or select individual needles."""
    deletion_method = input(
        "(b)ulk delete needles, (s)elect individually, or (r)eturn to menu:"
    )
    if deletion_method == "b":
        return needle_list
    if deletion_method == "s":
        needles_to_delete = []
        for needle in needle_list:
            option = input(f"Delete: {needle} (y/N):")
            if option == "y":
                needles_to_delete.append(needle)
        return needles_to_delete
    return []


def delete_needles_and_submit_mr(
    needles_list: List[str],
    needle_type: str,
    needles_git: git.Repo,
    needles_path: str,
    remote_branch: str,
    token: str,
    create_merge_request: bool,
) -> List[str]:
    """Deletes list of needles from needle repository, generates a commit and
    then a merge request.
    """
    if not needles_list:
        return []

    # Checkout remote branch and pull
    needles_git.git.checkout(remote_branch)
    git_cmd = git.cmd.Git(needles_path)
    git_cmd.pull("origin", remote_branch)

    # Create and checkout new branch (with unique epoch timestamp)
    branch_name = f"needles-cleanup/delete-{needle_type}-needles-{int(datetime.now().timestamp())}"
    needles_git.git.branch(branch_name)
    needles_git.git.checkout(branch_name)

    # Rebase and exit on conflicts
    output = needles_git.git.rebase(remote_branch)
    if re.search("conflicts", output, re.IGNORECASE):
        needles_git.git.rebase("--abort")
        raise Exception(
            "There was a conflict while rebasing before generating commit. \
            Exiting without deletion."
        )

    # Delete needles
    needles_to_delete = []
    untracked_files = []
    for needle in needles_list:
        needle_image = needle + ".png"
        needle_json = needle + ".json"
        if needle_image not in needles_git.untracked_files:
            needles_to_delete.append(needle_image)
        else:
            untracked_files.append(needle_image)
        if needle_json not in needles_git.untracked_files:
            needles_to_delete.append(needle_json)
        else:
            untracked_files.append(needle_image)
    needles_git.index.remove(needles_to_delete, working_tree=True)
    needles_git.index.commit(
        f"needles-cleanup script: deletion of {needle_type} needles"
    )

    if untracked_files:
        print("\nThe following untracked files were not deleted:")
        for untracked in untracked_files:
            print(untracked)

    print(
        f"\nCreated branch: {branch_name},  with a commit for the deletion of the needles"
    )

    if create_merge_request:
        # Push branch and create Gitlab MR from new branch

        if not token:
            raise gitlab.exceptions.GitlabAuthenticationError(
                "Authentication token must be set to create MRs"
            )

        # Push branch
        needles_git.git.push("--set-upstream", "origin", branch_name)

        # Create Merge Request
        gl = gitlab.Gitlab(GITLAB_URL, private_token=token)
        project = gl.projects.get(PROJECT_ID)
        title_string = f"Cleanup of {needle_type} needles"
        project.mergerequests.create(
            {
                "source_branch": branch_name,
                "target_branch": remote_branch,
                "title": title_string,
                "remove_source_branch": "true",
            }
        )
        print(f"Created MR for branch {branch_name}")

    # Return repository to main branch
    needles_git.git.checkout(remote_branch)

    return []


def commandline_interface(
    unused_needles: List[str],
    expired_needles: List[str],
    needles_git: git.Repo,
    needles_path: str,
    branch: str,
    expiry_date: str,
    token: str,
    create_merge_request: bool,
) -> None:
    """Commandline menu interface for deletion of needles."""
    while True:
        print("\nWhat would you like to do?")
        if unused_needles:
            print("[u] Delete unused needles")
        if expiry_date and expired_needles:
            print("[e] Delete expired needles")
        print("[q] Quit")
        command = input(":")
        if command == "u" and unused_needles:
            print("Deleting unused needles")
            needles_to_delete = select_for_deletion(unused_needles)
            unused_needles = delete_needles_and_submit_mr(
                needles_to_delete,
                "unused",
                needles_git,
                needles_path,
                branch,
                token,
                create_merge_request,
            )
        elif command == "e" and expiry_date and expired_needles:
            print("Deleting expired needles")
            needles_to_delete = select_for_deletion(expired_needles)
            delete_needles_and_submit_mr(
                needles_to_delete,
                "expired",
                needles_git,
                needles_path,
                branch,
                token,
                create_merge_request,
            )
        elif command == "q":
            break


def main():
    """Main program."""
    args = argument_parser().parse_args()

    missing = []
    badly_formatted_needle_names = []
    unused_needles = []
    expired_needles = []
    missing_json = []

    # List of openqa calls that make use of needles
    test_calls = ["assert_screen", "assert_and_click", "assert_recorded_sound"]

    needles_path = os.getcwd() + "/" + args.needles
    tests_path = os.getcwd() + "/" + args.tests + "/tests"

    # Get list of unique needles called from tests
    needle_calls = get_test_needles(tests_path, test_calls)
    # Get list of all the current needle names
    needle_list, missing_json = get_needle_list(needles_path)
    # Get a list of missing needles based on openqa test calls
    missing = check_missing_needles(needle_calls, needle_list)

    # Get lists of used/unused needles, and needles that have
    # badly formatted names, i.e. not <base-needle-name>-<datestamp>
    used_needles, unused_needles, badly_formatted_needle_names = find_needle_states(
        needle_calls, needle_list
    )

    # Create a list of expired needles based on a given expiry date
    if args.expiry_date:
        expired_needles = sorted(
            find_expired_needles(used_needles, needle_calls, args.expiry_date)
        )

    # Display a summary of needle states
    if args.summary:
        print_needle_summary(
            missing,
            badly_formatted_needle_names,
            unused_needles,
            expired_needles,
            missing_json,
            args.expiry_date,
        )
        sys.exit(0)

    # Obtain a git object for the needle repository
    needles_git = git.Repo(needles_path, search_parent_directories=True)

    # Commandline menu for needle deletion
    commandline_interface(
        unused_needles,
        expired_needles,
        needles_git,
        needles_path,
        args.branch,
        args.expiry_date,
        args.token,
        args.create_merge_request,
    )

    sys.exit(0)


try:
    main()
except Exception as e:
    print("Needle tool failed to run correctly")
    sys.stderr.write(f"ERROR: {e}\n")
    sys.exit(1)
