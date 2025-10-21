# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2024 Tomas Votava <thomasvotava@seznam.cz>

from pathlib import Path
from update_apps import (
    check_valid_values,
    VALID_CATEGORIES,
    VALID_FLAGS,
    VALID_DEFAULTS,
)

ID_List = dict[str, list[set[str]]]


def read_IDs_file(ids_file_location: str, main_ID_dict: ID_List) -> ID_List:
    options: dict[str, str] = {}
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
                    'Invalid format in file {0} on line {1}: "{2}"\n{3}'.format(
                        ids_file_location, line_index, line, "Too many columns"
                    )
                )
            if len(parameters) == 1:
                parameters = parameters[0].split("=")
                if len(parameters) == 1:
                    default_values = False
                    main_ID_dict[parameters[0]] = main_ID_dict.get(
                        parameters[0], [set(), set(), set()]
                    )
                    if "default-flags" not in set(options.keys()):
                        raise ValueError(
                            "Invalid value in file {0} on line {1}: {2}\n{3}".format(
                                ids_file_location,
                                line_index,
                                line,
                                "No default flags used",
                            )
                        )
                    main_ID_dict[parameters[0]][0].update(
                        set(options.get("default-flags", "").split(","))
                    )
                    main_ID_dict[parameters[0]][1].update(
                        set(options.get("default-categories", "").split(","))
                    )
                    continue
                if len(parameters) != 2:
                    raise ValueError(
                        'Invalid format in file {0} on line {1}: "{2}"\n{3}'.format(
                            ids_file_location,
                            line_index,
                            line,
                            "Invalid format of default values",
                        )
                    )
                if (
                    not default_values
                    or readable_line_index > len(VALID_DEFAULTS)
                    or parameters[0] in set(options.keys())
                ):
                    raise ValueError(
                        'Invalid value in file {0} on line {1}: "{2}"\n{3}'.format(
                            ids_file_location,
                            line_index,
                            line,
                            "Too many default options or default options"
                            "are not at the start of the file",
                        )
                    )
                if not check_valid_values(VALID_DEFAULTS, [parameters[0]]):
                    raise ValueError(
                        'Invalid value in file {0} on line {1}: "{2}"\n{3}'.format(
                            ids_file_location,
                            line_index,
                            line,
                            "Invalid default option",
                        )
                    )
                if parameters[0] == "default-flags" and not check_valid_values(
                    VALID_FLAGS, parameters[1].split(",")
                ):
                    raise ValueError(
                        'Invalid value in file {0} on line {1}: "{2}"\n{3}'.format(
                            ids_file_location, line_index, line, "Invalid flag"
                        )
                    )
                if parameters[0] == "default-categories" and not check_valid_values(
                    VALID_CATEGORIES, parameters[1].split(",")
                ):
                    raise ValueError(
                        'Invalid value in file {0} on line {1}: "{2}"\n{3}'.format(
                            ids_file_location, line_index, line, "Invalid category"
                        )
                    )

                options[parameters[0]] = parameters[1]
                continue
            default_values = False
            if (
                len(parameters) >= 2
                and parameters[1] == ""
                and "default-flags" not in set(options.keys())
            ):
                raise ValueError(
                    "Invalid value in file {0} on line {1}: {2}\n{3}".format(
                        ids_file_location, line_index, line, "No default flags used"
                    )
                )
            main_ID_dict[parameters[0]] = main_ID_dict.get(
                parameters[0], [set(), set(), set()]
            )
            if len(parameters) == 2 or len(parameters) > 2 and parameters[2] == "":
                main_ID_dict[parameters[0]][1].update(
                    set(options.get("default-categories", "").split(","))
                )

            for index in range(1, len(parameters)):
                main_ID_dict[parameters[0]][index - 1].update(
                    (set(parameters[index].split(",")))
                )

            if not check_valid_values(
                VALID_FLAGS, list(main_ID_dict[parameters[0]][0])
            ):
                raise ValueError(
                    'Invalid flag in file {0} on line {1}: "{2}"\n{3}'.format(
                        ids_file_location, line_index, line, "Invalid Flag"
                    )
                )

            if not check_valid_values(
                VALID_CATEGORIES, list(main_ID_dict[parameters[0]][1])
            ):
                raise ValueError(
                    'Invalid category in file {0} on line {1}: "{2}"\n{3}'.format(
                        ids_file_location, line_index, line, "Invalid category"
                    )
                )
    return main_ID_dict


def read_IDs_files(files: list[str]) -> dict[str, list[str]] | None:
    if len(files) == 0:
        print("No files were selected")
        return None
    main_ID_dict: ID_List = {}
    for file in files:
        path = Path(file)
        if not path.is_file():
            raise FileExistsError("File {0} doesn't exist".format(file))
        main_ID_dict = read_IDs_file(file, main_ID_dict)
    out_dict: dict[str, list[str]] = {}
    for id in main_ID_dict.keys():
        out_dict[id] = []
        for i in range(0, len(main_ID_dict[id])):
            if "" in main_ID_dict[id][i]:
                main_ID_dict[id][i].remove("")
            option_list = list(main_ID_dict[id][i])
            option_list.sort()
            out_dict[id].append(",".join(option_list))
    return out_dict
