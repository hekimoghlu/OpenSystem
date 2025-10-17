/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#include "NodeList.h"

#include <stdlib.h>

extern NodeList* NodeList_new(Node* parentNode)
{
    Node_ref(parentNode);

    NodeList* nodeList = (NodeList*)malloc(sizeof(NodeList));
    nodeList->parentNode = parentNode;
    nodeList->refCount = 0;
    return nodeList;
}

extern unsigned NodeList_length(NodeList* nodeList)
{
    /* Linear count from tail -- good enough for our purposes here */
    unsigned i = 0;
    NodeLink* n = nodeList->parentNode->childNodesTail;
    while (n) {
        n = n->prev;
        ++i;
    }

    return i;
}

extern Node* NodeList_item(NodeList* nodeList, unsigned index)
{
    unsigned length = NodeList_length(nodeList);
    if (index >= length)
        return NULL;

    /* Linear search from tail -- good enough for our purposes here */
    NodeLink* n = nodeList->parentNode->childNodesTail;
    unsigned i = 0;
    unsigned count = length - 1 - index;
    while (i < count) {
        ++i;
        n = n->prev;
    }
    return n->node;
}

extern void NodeList_ref(NodeList* nodeList)
{
    ++nodeList->refCount;
}

extern void NodeList_deref(NodeList* nodeList)
{
    if (--nodeList->refCount == 0) {
        Node_deref(nodeList->parentNode);
        free(nodeList);
    }
}
