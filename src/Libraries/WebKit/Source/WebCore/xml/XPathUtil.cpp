/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
#include "config.h"
#include "XPathUtil.h"

#include "ContainerNode.h"
#include "TextNodeTraversal.h"

namespace WebCore {
namespace XPath {
        
bool isRootDomNode(Node* node)
{
    return node && !node->parentNode();
}

String stringValue(Node* node)
{
    switch (node->nodeType()) {
        case Node::ATTRIBUTE_NODE:
        case Node::PROCESSING_INSTRUCTION_NODE:
        case Node::COMMENT_NODE:
        case Node::TEXT_NODE:
        case Node::CDATA_SECTION_NODE:
            return node->nodeValue();
        default:
            if (isRootDomNode(node) || node->isElementNode())
                return TextNodeTraversal::contentsAsString(*node);
    }
    return String();
}

bool isValidContextNode(Node& node)
{
    switch (node.nodeType()) {
        case Node::ATTRIBUTE_NODE:
        case Node::CDATA_SECTION_NODE:
        case Node::COMMENT_NODE:
        case Node::DOCUMENT_NODE:
        case Node::ELEMENT_NODE:
        case Node::PROCESSING_INSTRUCTION_NODE:
        case Node::TEXT_NODE:
            return true;
        case Node::DOCUMENT_FRAGMENT_NODE:
        case Node::DOCUMENT_TYPE_NODE:
            return false;
    }
    ASSERT_NOT_REACHED();
    return false;
}

}
}
