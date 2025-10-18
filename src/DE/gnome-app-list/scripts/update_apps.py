# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2024 Tomas Votava <thomasvotava@seznam.cz>

import json
import http.client
import time
import argparse
from pathlib import Path


FLATHUB_URL = "flathub.org"
JSON_FLATHUB_PATH = "/api/v2/quality-moderation/" "passing-apps?page={0}&page_size=25"
LOCAL_FLATHUB_FILE = "data/flathub-apps.txt"


GNOME_URL = "gitlab.gnome.org"
JSON_GNOME_PATH = "/Teams/Circle/-/raw/main/data/apps.json"
LOCAL_GNOME_FILE = "data/gnome-apps.txt"

VALID_ARGUMENTS: set[str] = {"gnome", "flathub"}

# List of categories taken from:
# https://specifications.freedesktop.org/menu-spec/latest/apa.html
# on 27. 06. 2024
VALID_CATEGORIES: set[str] = {
    "Audio",
    "Video",
    "AudioVideo",
    "Development",
    "Education",
    "Game",
    "Graphics",
    "Network",
    "Office",
    "Science",
    "Settings",
    "System",
    "Utility",
    #   Custom categories
    "Featured",
}

VALID_FLAGS: set[str] = {"popular", "skip", "featured", ""}

VALID_DEFAULTS: set[str] = {"default-flags", "default-categories"}


def check_valid_values(valid_set: set[str], check_list: list[str]) -> bool:
    if len(check_list) == 0 or (len(check_list) == 1 and check_list[0] == ""):
        return True
    for string in check_list:
        if string not in valid_set:
            print("{0} is not valid option".format(string))
            return False
    return True


def get_flathub_ids() -> list[str] | None:
    flathub_IDs: list[str] = []
    index: int = 1
    connection = http.client.HTTPSConnection(FLATHUB_URL)
    #    Emulation of do while cycle
    while True:
        connection.request("GET", JSON_FLATHUB_PATH.format(index))
        response = connection.getresponse()
        if response.reason != "OK":
            connection.close()
            return None
        unprocessed_data = response.read().decode("utf-8")
        data_in = json.loads(unprocessed_data)
        flathub_IDs.extend(data_in["apps"])
        if data_in["pagination"]["total_pages"] == index:
            break
        index += 1
    connection.close()
    return flathub_IDs


def save_IDs(
    content: list[str], timestamp: time.struct_time, ids_file_location: str
) -> None:
    options: list[tuple[str, str]] = []
    env: dict[str, list[str]] = {}
    old_file = Path(ids_file_location)
    content.sort()
    if old_file.is_file():
        options, env = read_existing_IDs(ids_file_location)
    output: str = ""
    output += "# Generated on " + time.asctime(timestamp) + "\n"
    output += """# SPDX-License-Identifier: LGPL-2.1-or-later

"""
    for option, value in options:
        output += option + "=" + value + "\n"
    for id in content:
        out = id
        additional_data = env.get(id, [])
        for data in additional_data:
            out += "\t" + data
        output += out + "\n"
    with open(ids_file_location, "w") as file:
        file.write(output)
        return


def check_options(options_list: list[tuple[str, str]], option: str) -> bool:
    for active_option, _ in options_list:
        if active_option == option:
            return False
    return True


def read_existing_IDs(
    ids_file_location: str,
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    enviroment: dict[str, list[str]] = {}
    options: list[tuple[str, str]] = []
    with open(ids_file_location, "r") as file:
        readable_line_index = 0
        default_values = True
        line_index = 0
        for line in file:
            line_index += 1
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            readable_line_index += 1
            parameters = line.split("\t")
            if len(parameters) > 3:
                raise ValueError(
                    "Invalid value in line {0}: {1}\n{2}".format(
                        line_index, line, "Too many columns"
                    )
                )
            if len(parameters) == 1:
                parameters = parameters[0].split("=")
                if len(parameters) == 1:
                    if check_options(options, "default-flags"):
                        raise ValueError(
                            "Invalid value in line {0}: {1}\n{2}".format(
                                line_index, line, "No default flags used"
                            )
                        )
                    enviroment[parameters[0]] = []
                    default_values = False
                    continue
                if len(parameters) != 2:
                    raise ValueError(
                        "Invalid value in line {0}: {1}\n{2}".format(
                            line_index, line, "Invalid format of default-flags"
                        )
                    )
                if (
                    not default_values
                    or readable_line_index > len(VALID_DEFAULTS)
                    or not check_options(options, parameters[0])
                ):
                    raise ValueError(
                        "Invalid value in line {0}: {1}\n{2}".format(
                            line_index,
                            line,
                            "Too many default options or default "
                            "options are not at the start of the file",
                        )
                    )
                if not check_valid_values(VALID_DEFAULTS, [parameters[0]]):
                    raise ValueError(
                        "Invalid value in line {0}: {1}\n{2}".format(
                            line_index, line, "Invalid default option"
                        )
                    )
                if parameters[0] == "default-flags" and not check_valid_values(
                    VALID_FLAGS, parameters[1].split(",")
                ):
                    raise ValueError(
                        "Invalid value in line {0}: {1}\n{2}".format(
                            line_index, line, "Invalid Flag"
                        )
                    )
                if parameters[0] == "default-categories" and not check_valid_values(
                    VALID_CATEGORIES, parameters[1].split(",")
                ):
                    raise ValueError(
                        "Invalid value in line {0}: {1}\n{2}".format(
                            line_index, line, "Invalid category"
                        )
                    )
                options.append((parameters[0], parameters[1]))
                continue
            default_values = False
            enviroment[parameters[0]] = parameters
            if parameters[1] == "" and check_options(options, "default-flags"):
                raise ValueError(
                    "Invalid value in line {0}: {1}\n{2}".format(
                        line_index, line, "No default flags used"
                    )
                )
            parameters.pop(0)
            if len(parameters) > 0 and not check_valid_values(
                VALID_FLAGS, parameters[0].split(",")
            ):
                raise ValueError(
                    "Invalid value in line {0}: {1}\n{2}".format(
                        line_index, line, "Invalid Flag"
                    )
                )
            if len(parameters) > 1 and not check_valid_values(
                VALID_CATEGORIES, parameters[1].split(",")
            ):
                raise ValueError(
                    "Invalid value in line {0}: {1}\n{2}".format(
                        line_index, line, "Invalid Flag"
                    )
                )
    return (options, enviroment)


def get_gnome_ids() -> list[str] | None:
    gnome_IDs: list[str] = []
    connection = http.client.HTTPSConnection(GNOME_URL)
    connection.request("GET", JSON_GNOME_PATH)
    response = connection.getresponse()
    if response.reason != "OK":
        connection.close()
        return None
    unprocessed_data = response.read().decode("utf-8")
    data_in = json.loads(unprocessed_data)
    for component in data_in:
        gnome_IDs.append(component["app_id"])
    connection.close()
    return gnome_IDs


def update_gnome() -> None:
    new_ids = get_gnome_ids()
    if new_ids is None:
        print("Couldn't load new GNOME apps")
        return
    save_IDs(new_ids, time.gmtime(), LOCAL_GNOME_FILE)


def update_flathub() -> None:
    new_ids = get_flathub_ids()
    if new_ids is None:
        print("Couldn't load new Flathub apps")
        return
    save_IDs(new_ids, time.gmtime(), LOCAL_FLATHUB_FILE)


UPDATED_STRING = "{0} apps updated\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Updates ID files.",
        epilog="""Choosing no option will update both
    GNOME and Flathub apps list""",
    )
    parser.add_argument(
        "options",
        type=str,
        nargs="*",
        choices=["flathub", "gnome"],
        help="""specifies which files to update, use \'gnome\' to
        update GNOME apps and/or \'flathub\'
        to update Flathub apps""",
    )
    args = parser.parse_args()
    if len(args.options) == 0:
        print("Updating Flathub apps and GNOME apps...")
        update_flathub()
        update_gnome()
        print(UPDATED_STRING.format("Flathub and GNOME"))
        return 0

    if "flathub" in set(args.options):
        print("Updating Flathub apps...")
        update_flathub()
        print(UPDATED_STRING.format("Flathub"))

    if "gnome" in set(args.options):
        print("Updating GNOME apps...")
        update_gnome()
        print(UPDATED_STRING.format("GNOME"))

    return 0


if __name__ == "__main__":
    main()
