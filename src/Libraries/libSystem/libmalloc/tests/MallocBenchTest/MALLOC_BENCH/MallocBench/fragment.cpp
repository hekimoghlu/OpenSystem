/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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
#include "CPUCount.h"
#include "fragment.h"
#include <stdlib.h>
#include <strings.h>

#include "mbmalloc.h"

namespace {

class Node {
public:
    void* operator new(size_t size)
    {
        return mbmalloc(size);
    }

    void operator delete(void* p, size_t size)
    {
        mbfree(p, size);
    }

    Node()
        : m_next(0)
        , m_payload()
    {
    }

    Node(Node* next)
        : m_next(next)
        , m_payload()
    {
    }
    
    Node* next() { return m_next; }
    
    void validate()
    {
        for (size_t i = 0; i < sizeof(m_payload); ++i) {
            if (m_payload[i])
                abort();
        }
    }

private:
    Node* m_next;
    char m_payload[32 - sizeof(Node*)];
};

} // namespace

void validate(Node* head)
{
    for (Node* node = head; node; node = node->next())
        node->validate();
}

void benchmark_fragment(CommandLine& commandLine)
{
    size_t nodeCount = 128 * 1024;
    if (commandLine.isParallel())
        nodeCount /= cpuCount();
    size_t replaceCount = nodeCount / 4;
    size_t times = 25;

    srandom(0); // For consistency between runs.

    for (size_t i = 0; i < times; ++i) {
        Node** nodes = static_cast<Node**>(mbmalloc(nodeCount * sizeof(Node*)));
        for (size_t i = 0; i < nodeCount; ++i)
            nodes[i] = new Node;

        for (size_t i = 0; i < replaceCount; ++i) {
            size_t node = random() % nodeCount;

            delete nodes[node];
            nodes[node] = new Node;
        }

        for (size_t node = 0; node < nodeCount; ++node)
            delete nodes[node];
        mbfree(nodes, nodeCount * sizeof(Node*));
    }
}

void benchmark_fragment_iterate(CommandLine& commandLine)
{
    size_t nodeCount = 512 * 1024;
    size_t times = 20;
    if (commandLine.isParallel())
        nodeCount /= cpuCount();
    size_t replaceCount = nodeCount / 4;

    srandom(0); // For consistency between runs.

    Node** nodes = static_cast<Node**>(mbmalloc(nodeCount * sizeof(Node*)));
    for (size_t i = 0; i < nodeCount; ++i)
        nodes[i] = new Node;

    Node* head = 0;
    for (size_t i = 0; i < replaceCount; ++i) {
        size_t node = random() % nodeCount;

        delete nodes[node];
        nodes[node] = 0;
        head = new Node(head);
    }
    
    for (size_t i = 0; i < times; ++i)
        validate(head);

    for (Node* next ; head; head = next) {
        next = head->next();
        delete head;
    }

    for (size_t node = 0; node < nodeCount; ++node) {
        if (!nodes[node])
            continue;
        delete nodes[node];
    }
    mbfree(nodes, nodeCount * sizeof(Node*));
}
