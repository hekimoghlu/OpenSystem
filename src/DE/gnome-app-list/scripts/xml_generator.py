# SPDX-License-Identifier: LGPL-2.1-or-later
# SPDX-FileCopyrightText: 2024 Tomas Votava <thomasvotava@seznam.cz>

import argparse
import xml.etree.ElementTree as ET
from connect_ids import read_IDs_files

POPULAR: ET.Element = ET.Element("kudos")
kudo: ET.Element = ET.SubElement(POPULAR, "kudo")
kudo.text = "GnomeSoftware::popular"

FEATURED: ET.Element = ET.Element("custom")
value: ET.Element = ET.SubElement(
    FEATURED, "value", {"key": "GnomeSoftware::FeatureTile"}
)
value.text = "True"


ID_List = dict[str, list[str]]

XML_Location: str = "org.gnome.App-list.xml"

FLAGS_DICT: dict[str, ET.Element] = {
    "popular": POPULAR,
    "featured": FEATURED,
}

COMPONENT_ATTR: dict[str, str] = {"merge": "append"}


def generate_xml(path: str, complete_list: ID_List) -> None:
    output: str = ""
    output += """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<!-- SPDX-License-Identifier: LGPL-2.1-or-later -->\n"""
    components: ET.Element = ET.Element("components")
    tree: ET.ElementTree = ET.ElementTree(components)
    for id, flags in complete_list.items():
        if "skip" in set(flags[0].split(",")):
            continue
        component: ET.Element = ET.SubElement(components, "component", COMPONENT_ATTR)
        id_elem: ET.Element = ET.SubElement(component, "id")
        id_elem.text = id
        if len(flags[0]) != 0:
            for flag in flags[0].split(","):
                if flag != "":
                    component.append(FLAGS_DICT[flag])
        if len(flags[1]) != 0:
            categories: ET.Element = ET.SubElement(component, "categories")
            for category in flags[1].split(","):
                category_elem: ET.Element = ET.SubElement(categories, "category")
                category_elem.text = category
    ET.indent(tree)
    with open(path, "w") as file:
        file.write(output)
        file.write(ET.tostring(components, "utf-8").decode("utf-8"))
        return


def generate_xml_from_files(path: str, files: list[str]) -> None:
    ids_dict = read_IDs_files(files)
    if ids_dict is None:
        raise Exception("No IDs were loaded from files")
    generate_xml(path, ids_dict)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generates org.gnome.App-list.xml file from given ID files"
    )
    parser.add_argument(
        "file", type=str, nargs="+", help="ID files to create xml file from"
    )
    args = parser.parse_args()
    files = args.file
    generate_xml_from_files(XML_Location, files)
    return 0


if __name__ == "__main__":
    main()
