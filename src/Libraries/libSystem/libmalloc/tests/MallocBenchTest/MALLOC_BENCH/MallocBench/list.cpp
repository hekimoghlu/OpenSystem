/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "list.h"
#include <stdlib.h>
#include <strings.h>

#include "mbmalloc.h"

namespace {

struct Node {
    void* operator new(size_t size)
    {
        return mbmalloc(size);
    }

    void operator delete(void* p, size_t size)
    {
        mbfree(p, size);
    }

    Node(Node* next, size_t payloadSize)
        : m_refCount(1)
        , m_next(next)
        , m_payload(static_cast<char*>(mbmalloc(payloadSize)))
        , m_payloadSize(payloadSize)
    {
        if (m_next)
            m_next->ref();
        bzero(m_payload, payloadSize);
    }

    ~Node()
    {
        if (m_next)
            m_next->deref();
        mbfree(m_payload, m_payloadSize);
    }

    void ref()
    {
        ++m_refCount;
    }

    void deref()
    {
        if (m_refCount == 1)
            delete this;
        else
            --m_refCount;
    }

    Node* takeNext()
    {
        Node* tmp = m_next;
        m_next = 0;
        return tmp;
    }
    
    bool validate()
    {
        if (m_payload[0])
            return false;
        return true;
    }

    unsigned m_refCount;
    Node* m_next;
    char* m_payload;
    size_t m_payloadSize;
};

} // namespace

void benchmark_list_allocate(CommandLine& commandLine)
{
    Node* head = 0;
    size_t times = 70;
    size_t nodes = 32 * 1024;
    if (commandLine.isParallel()) {
        nodes /= cpuCount();
        times *= 2;
    }
    
    for (size_t time = 0; time < times; ++time) {
        // Construct a list of nodes.
        for (size_t node = 0; node < nodes; ++node) {
            Node* oldHead = head;
            head = new Node(oldHead, (nodes & (64 - 1)) | 1);
            if (oldHead)
                oldHead->deref();
        }

        // Tear down the list.
        while (head) {
            Node* tmp = head->takeNext();
            head->deref();
            head = tmp;
        }
    }
}

void benchmark_list_traverse(CommandLine& commandLine)
{
    Node* head = 0;
    size_t times = 1 * 1024;
    size_t nodes = 32 * 1024;
    if (commandLine.isParallel()) {
        nodes /= cpuCount();
        times *= 4;
    }

    // Construct a list of nodes.
    for (size_t node = 0; node < nodes; ++node) {
        Node* oldHead = head;
        head = new Node(oldHead, (nodes & (64 - 1)) | 1);
        if (oldHead)
            oldHead->deref();
    }

    // Validate the list.
    for (size_t time = 0; time < times; ++time) {
        for (Node* node = head; node; node = node->m_next) {
            if (!node->validate())
                abort();
        }
    }

    // Tear down the list.
    while (head) {
        Node* tmp = head->takeNext();
        head->deref();
        head = tmp;
    }
}
