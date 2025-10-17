/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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
#ifndef __JSON_H__
#define __JSON_H__

#include <string.h>

#include <string>
#include <map>
#include <sstream>
#include <vector>

namespace json {

enum class NodeValueType {
    Default,
    String,
    Array,
    Map,
    RawValue,
};

struct Node
{
    NodeValueType               type = NodeValueType::Default;
    std::string                 value;
    std::map<std::string, Node> map;
    std::vector<Node>           array;

    inline Node()
    : type(NodeValueType::Default), value(), map(), array() { }

    inline Node(std::string string)
    : type(NodeValueType::String), value(string), map(), array() { }

    inline Node(const char *string) : Node(std::string{string}) { }

    inline Node(bool b)
    : type(NodeValueType::RawValue), value(b ? "true" : "false")
    , map(), array() { }

    inline Node(int64_t i64)
    : type(NodeValueType::RawValue), value(), map(), array()
    {
        std::ostringstream os{};
        os << i64;
        value = os.str();
    }

    inline Node(uint64_t u64)
    : type(NodeValueType::RawValue), value(), map(), array()
    {
        std::ostringstream os{};
        os << u64;
        value = os.str();
    }

    // remove node* initializers to avoid implicit conversion to value types
    inline Node(const Node*) = delete;
    inline Node(Node*) = delete;
};

static inline Node makeNode(std::string value) {
    Node node;
    node.value = value;
    return node;
}

} // namespace json


#endif // __JSON_H__
