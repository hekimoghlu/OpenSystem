/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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

#include <wtf/Atomics.h>
#include <wtf/Noncopyable.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

// This a simple single consumer, multiple producer Bag data structure.

template<typename T>
class LocklessBag final {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(LocklessBag);
public:
    struct Node {
        WTF_MAKE_FAST_ALLOCATED;
    public:
        T data;
        Node* next;
    };

    LocklessBag()
        : m_head(nullptr)
    {
    }

    enum PushResult { Empty, NonEmpty };
    PushResult add(T&& element)
    {
        Node* newNode = new Node();
        newNode->data = std::forward<T>(element);

        Node* oldHead;
        m_head.transaction([&] (Node*& head) {
            oldHead = head;
            newNode->next = head;
            head = newNode;
            return true;
        });

        return oldHead == nullptr ? Empty : NonEmpty;
    }

    // CONSUMER FUNCTIONS: Everything below here is only safe to call from the consumer thread.

    // This function is actually safe to call from more than one thread, but ONLY if no thread can call consumeAll.
    void iterate(NOESCAPE const Invocable<void(const T&)> auto& func)
    {
        Node* node = m_head.load();
        while (node) {
            func(node->data);
            node = node->next;
        }
    }

    void consumeAll(NOESCAPE const Invocable<void(T&&)> auto& func)
    {
        consumeAllWithNode([&] (T&& data, Node* node) {
            func(WTFMove(data));
            delete node;
        });
    }

    void consumeAllWithNode(NOESCAPE const Invocable<void(T&&, Node*)> auto& func)
    {
        Node* node = m_head.exchange(nullptr);
        while (node) {
            Node* oldNode = node;
            node = node->next;
            func(WTFMove(oldNode->data), oldNode);
        }
    }

    ~LocklessBag()
    {
        consumeAll([] (T&&) { });
    }

private:
    Atomic<Node*> m_head;
};
    
} // namespace WTF
