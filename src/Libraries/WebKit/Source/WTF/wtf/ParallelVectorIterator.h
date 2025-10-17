/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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

#include <wtf/FastMalloc.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>

namespace WTF {

template<typename VectorType>
class ParallelVectorIterator {
    WTF_MAKE_NONCOPYABLE(ParallelVectorIterator);
    WTF_MAKE_FAST_ALLOCATED;
public:
    ParallelVectorIterator(VectorType& vector, size_t shardSize)
        : m_vector(vector)
        , m_shardSize(shardSize)
    {
    }

    template<typename Functor>
    ALWAYS_INLINE void iterate(const Functor& functor)
    {
        for (;;) {
            size_t begin;
            size_t end;
            {
                Locker locker { m_lock };
                begin = m_next;
                if (begin == m_vector.size())
                    return;
                if (m_vector.size() - begin < m_shardSize)
                    end = m_vector.size();
                else
                    end = begin + m_shardSize;
                RELEASE_ASSERT(end <= m_vector.size());
                RELEASE_ASSERT(begin < end);
                m_next = end;
            }

            for (size_t i = begin; i < end; ++i)
                functor(m_vector[i]);
        }
    }
private:
    Lock m_lock;
    VectorType& m_vector WTF_GUARDED_BY_LOCK(m_lock);
    size_t m_shardSize WTF_GUARDED_BY_LOCK(m_lock);
    size_t m_next WTF_GUARDED_BY_LOCK(m_lock) { 0 };
};

} // namespace WTF

using WTF::ParallelVectorIterator;
