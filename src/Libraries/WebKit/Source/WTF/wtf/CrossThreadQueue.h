/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 14, 2023.
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

#include <limits>
#include <wtf/Assertions.h>
#include <wtf/Condition.h>
#include <wtf/Deque.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>

namespace WTF {

template<typename DataType>
class CrossThreadQueue final {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(CrossThreadQueue);
public:
    CrossThreadQueue() = default;

    void append(DataType&&);

    DataType waitForMessage();
    std::optional<DataType> tryGetMessage();

    void kill();
    bool isKilled() const;
    bool isEmpty() const;

private:
    mutable Lock m_lock;
    Deque<DataType> m_queue WTF_GUARDED_BY_LOCK(m_lock);
    Condition m_condition;
    bool m_killed WTF_GUARDED_BY_LOCK(m_lock) { false };
};

template<typename DataType>
void CrossThreadQueue<DataType>::append(DataType&& message)
{
    Locker locker { m_lock };
    ASSERT(!m_killed);
    m_queue.append(WTFMove(message));
    m_condition.notifyOne();
}

template<typename DataType>
DataType CrossThreadQueue<DataType>::waitForMessage()
{
    Locker locker { m_lock };

    auto found = m_queue.end();
    while (!m_killed && found == m_queue.end()) {
        found = m_queue.begin();
        if (found != m_queue.end())
            break;

        m_condition.wait(m_lock);
    }
    if (m_killed)
        return { };

    return m_queue.takeFirst();
}

template<typename DataType>
std::optional<DataType> CrossThreadQueue<DataType>::tryGetMessage()
{
    Locker locker { m_lock };

    if (m_queue.isEmpty())
        return { };

    return m_queue.takeFirst();
}

template<typename DataType>
void CrossThreadQueue<DataType>::kill()
{
    Locker locker { m_lock };
    m_killed = true;
    m_condition.notifyAll();
}

template<typename DataType>
bool CrossThreadQueue<DataType>::isKilled() const
{
    Locker locker { m_lock };
    return m_killed;
}

template<typename DataType>
bool CrossThreadQueue<DataType>::isEmpty() const
{
    Locker locker { m_lock };
    return m_queue.isEmpty();
}

} // namespace WTF

using WTF::CrossThreadQueue;
