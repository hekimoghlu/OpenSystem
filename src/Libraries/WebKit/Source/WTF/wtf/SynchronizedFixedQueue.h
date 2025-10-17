/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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

#include <wtf/Condition.h>
#include <wtf/Deque.h>
#include <wtf/Lock.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WTF {

template<typename T, size_t BufferSize>
class SynchronizedFixedQueue final : public ThreadSafeRefCounted<SynchronizedFixedQueue<T, BufferSize>> {
public:
    static Ref<SynchronizedFixedQueue> create()
    {
        return adoptRef(*new SynchronizedFixedQueue());
    }

    void open()
    {
        Locker locker { m_lock };
        if (m_open)
            return;

        // Restore the queue to its original state.
        m_open = true;
        m_queue.clear();
    }

    void close()
    {
        Locker locker { m_lock };
        if (!m_open)
            return;

        // Wake all the sleeping threads up with a closing state.
        m_open = false;
        m_condition.notifyAll();
    }

    bool isOpen()
    {
        Locker locker { m_lock };
        return m_open;
    }

    bool enqueue(const T& value)
    {
        Locker locker { m_lock };

        // Wait for an empty place to be available in the queue.
        m_condition.wait(m_lock, [this] {
            assertIsHeld(m_lock);
            return !m_open || m_queue.size() < BufferSize;
        });

        // The queue is closing, exit immediately.
        if (!m_open)
            return false;

        // Add the item in the queue.
        m_queue.append(value);

        // Notify the other threads that an item was added to the queue.
        m_condition.notifyAll();
        return true;
    }

    bool dequeue(T& value)
    {
        Locker locker { m_lock };

        // Wait for an item to be added.
        m_condition.wait(m_lock, [this] {
            assertIsHeld(m_lock);
            return !m_open || m_queue.size();
        });

        // The queue is closing, exit immediately.
        if (!m_open)
            return false;

        // Get a copy from m_queue.first and then remove it.
        value = m_queue.first();
        m_queue.removeFirst();

        // Notify the other threads that an item was removed from the queue.
        m_condition.notifyAll();
        return true;
    }

private:
    SynchronizedFixedQueue()
    {
        static_assert(!((BufferSize - 1) & BufferSize), "BufferSize must be power of 2.");
    }

    Lock m_lock;
    Condition m_condition;

    bool m_open WTF_GUARDED_BY_LOCK(m_lock) { true };
    Deque<T, BufferSize> m_queue WTF_GUARDED_BY_LOCK(m_lock);
};

}

using WTF::SynchronizedFixedQueue;
