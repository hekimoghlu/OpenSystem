/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#pragma once

#include "NodeTraversal.h"
#include "Text.h"

namespace WTF {
class StringBuilder;
}

namespace WebCore {
namespace TextNodeTraversal {

// First text child of the node.
Text* firstChild(const Node&);
Text* firstChild(const ContainerNode&);

// First text descendant of the node.
Text* firstWithin(const Node&);
Text* firstWithin(const ContainerNode&);

// Pre-order traversal skipping non-text nodes.
Text* next(const Node&);
Text* next(const Node&, const Node* stayWithin);
Text* next(const Text&);
Text* next(const Text&, const Node* stayWithin);

// Next text sibling.
Text* nextSibling(const Node&);

// Concatenated text contents of a subtree.
String contentsAsString(const Node&);
String contentsAsString(const ContainerNode&);
void appendContents(const ContainerNode&, StringBuilder& result);
String childTextContent(const ContainerNode&);

}

namespace TextNodeTraversal {

template <class NodeType>
inline Text* firstTextChildTemplate(NodeType& current)
{
    for (auto* node = current.firstChild(); node; node = node->nextSibling()) {
        if (auto* text = dynamicDowncast<Text>(*node))
            return text;
    }
    return nullptr;
}
inline Text* firstChild(const Node& current) { return firstTextChildTemplate(current); }
inline Text* firstChild(const ContainerNode& current) { return firstTextChildTemplate(current); }

template <class NodeType>
inline Text* firstTextWithinTemplate(NodeType& current)
{
    for (auto* node = current.firstChild(); node; node = NodeTraversal::next(*node, &current)) {
        if (auto* text = dynamicDowncast<Text>(*node))
            return text;
    }
    return nullptr;
}
inline Text* firstWithin(const Node& current) { return firstTextWithinTemplate(current); }
inline Text* firstWithin(const ContainerNode& current) { return firstTextWithinTemplate(current); }

template <class NodeType>
inline Text* traverseNextTextTemplate(NodeType& current)
{
    for (auto* node = NodeTraversal::next(current); node; node = NodeTraversal::next(*node)) {
        if (auto* text = dynamicDowncast<Text>(*node))
            return text;
    }
    return nullptr;
}
inline Text* next(const Node& current) { return traverseNextTextTemplate(current); }
inline Text* next(const Text& current) { return traverseNextTextTemplate(current); }

template <class NodeType>
inline Text* traverseNextTextTemplate(NodeType& current, const Node* stayWithin)
{
    for (auto* node = NodeTraversal::next(current, stayWithin); node; node = NodeTraversal::next(*node, stayWithin)) {
        if (auto* text = dynamicDowncast<Text>(*node))
            return text;
    }
    return nullptr;
}
inline Text* next(const Node& current, const Node* stayWithin) { return traverseNextTextTemplate(current, stayWithin); }
inline Text* next(const Text& current, const Node* stayWithin) { return traverseNextTextTemplate(current, stayWithin); }

inline Text* nextSibling(const Node& current)
{
    for (auto* node = current.nextSibling(); node; node = node->nextSibling()) {
        if (auto* text = dynamicDowncast<Text>(*node))
            return text;
    }
    return nullptr;
}

} // namespace TextNodeTraversal
} // namespace WebCore
