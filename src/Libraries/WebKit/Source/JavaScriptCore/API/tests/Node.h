/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 18, 2025.
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

typedef struct __Node Node;
typedef struct __NodeLink NodeLink;

struct __NodeLink {
    Node* node;
    NodeLink* prev;
};

struct __Node {
    unsigned refCount;
    const char* nodeType;
    NodeLink* childNodesTail;
};

extern Node* Node_new(void);
extern void Node_ref(Node* node);
extern void Node_deref(Node* node);
extern void Node_appendChild(Node* node, Node* child);
extern void Node_removeChild(Node* node, Node* child);
extern void Node_replaceChild(Node* node, Node* newChild, Node* oldChild);
