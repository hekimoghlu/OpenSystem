/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#include "Node.h"

#include <stddef.h>
#include <stdlib.h>

Node* Node_new(void)
{
    Node* node = (Node*)malloc(sizeof(Node));
    node->refCount = 0;
    node->nodeType = "Node";
    node->childNodesTail = NULL;
    
    return node;
}

void Node_appendChild(Node* node, Node* child)
{
    Node_ref(child);
    NodeLink* nodeLink = (NodeLink*)malloc(sizeof(NodeLink));
    nodeLink->node = child;
    nodeLink->prev = node->childNodesTail;
    node->childNodesTail = nodeLink;
}

void Node_removeChild(Node* node, Node* child)
{
    /* Linear search from tail -- good enough for our purposes here */
    NodeLink* current;
    NodeLink** currentHandle;
    for (currentHandle = &node->childNodesTail, current = *currentHandle; current; currentHandle = &current->prev, current = *currentHandle) {
        if (current->node == child) {
            Node_deref(current->node);
            *currentHandle = current->prev;
            free(current);
            break;
        }
    }
}

void Node_replaceChild(Node* node, Node* newChild, Node* oldChild)
{
    /* Linear search from tail -- good enough for our purposes here */
    NodeLink* current;
    for (current = node->childNodesTail; current; current = current->prev) {
        if (current->node == oldChild) {
            Node_deref(current->node);
            current->node = newChild;
        }
    }
}

void Node_ref(Node* node)
{
    ++node->refCount;
}

void Node_deref(Node* node)
{
    if (--node->refCount == 0)
        free(node);
}
