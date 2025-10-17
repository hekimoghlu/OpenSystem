/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 6, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef __JSON_WRITER_H__
#define __JSON_WRITER_H__

#include <iostream>

#include "JSON.h"

namespace json {

static inline std::string hex(uint64_t value) {
    char buff[64];
    snprintf(buff, sizeof(buff), "0x%llX", value);
    return buff;
}

static inline std::string hex4(uint64_t value) {
    char buff[64];
    snprintf(buff, sizeof(buff), "0x%04llX", value);
    return buff;
}

static inline std::string hex8(uint64_t value) {
    char buff[64];
    snprintf(buff, sizeof(buff), "0x%08llX", value);
    return buff;
}

static inline std::string unpaddedDecimal(uint64_t value) {
    char buff[64];
    snprintf(buff, sizeof(buff), "%llu", value);
    return buff;
}

static inline std::string decimal(uint64_t value) {
    char buff[64];
    snprintf(buff, sizeof(buff), "%02llu", value);
    return buff;
}

static inline void indentBy(uint32_t spaces, std::ostream& out) {
    for (uint32_t i=0; i < spaces; ++i) {
        out << " ";
    }
}

static inline void printJSON(const Node& node, uint32_t indent = 0, std::ostream& out = std::cout)
{
    if ( !node.map.empty() ) {
        out << "{";
        bool needComma = false;
        for (const auto& entry : node.map) {
            if ( needComma )
                out << ",";
            out << "\n";
            indentBy(indent+2, out);
            out << "\"" << entry.first << "\": ";
            printJSON(entry.second, indent+2, out);
            needComma = true;
        }
        out << "\n";
        indentBy(indent, out);
        out << "}";
    }
    else if ( !node.array.empty() ) {
        out << "[";
        bool needComma = false;
        for (const auto& entry : node.array) {
            if ( needComma )
                out << ",";
            out << "\n";
            indentBy(indent+2, out);
            printJSON(entry, indent+2, out);
            needComma = true;
        }
        out << "\n";
        indentBy(indent, out);
        out << "]";
    }
    else {
        auto &value = node.value;
        switch (node.type) {
        case NodeValueType::Default:
        case NodeValueType::String:
            if (value.find('"') == std::string::npos) {
                out << "\"" << value << "\"";
            } else {
                std::string escapedString;
                escapedString.reserve(value.size());
                for (char c : value) {
                    if (c == '"')
                        escapedString += '\\';
                    escapedString += c;
                }
                out << "\"" << escapedString << "\"";
            }
            break;
        case NodeValueType::RawValue:
            out << value;
            break;
        case NodeValueType::Array:
        case NodeValueType::Map:
            // handled earlier
            break;
        }
    }
    if ( indent == 0 )
        out << "\n";
}

static inline void streamArrayBegin(bool &needsComma, std::ostream& out = std::cout)
{
    out << "[";
    needsComma = false;
}

static inline void streamArrayNode(bool &needsComma, Node &node, std::ostream& out = std::cout)
{
    if (needsComma)
        out << ",";
    out << "\n";
    indentBy(2, out);
    printJSON(node, 2, out);
    needsComma = true;
}

static inline void streamArrayEnd(bool &needsComma, std::ostream& out = std::cout)
{
    if (needsComma)
        out << "\n";
    out << "]\n";
}

} // namespace json


#endif // __JSON_WRITER_H__
