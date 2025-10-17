/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 24, 2022.
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

#include "GCSegmentedArray.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

template <typename T>
GCSegmentedArray<T>::GCSegmentedArray()
    : m_top(0)
    , m_numberOfSegments(0)
{
    m_segments.push(GCArraySegment<T>::create());
    m_numberOfSegments++;
}

template <typename T>
GCSegmentedArray<T>::~GCSegmentedArray()
{
    ASSERT(m_numberOfSegments == 1);
    ASSERT(m_segments.size() == 1);
    GCArraySegment<T>::destroy(m_segments.removeHead());
    m_numberOfSegments--;
    ASSERT(!m_numberOfSegments);
    ASSERT(!m_segments.size());
}

template <typename T>
void GCSegmentedArray<T>::clear()
{
    if (!m_segments.head())
        return;
    GCArraySegment<T>* next;
    for (GCArraySegment<T>* current = m_segments.head(); current->next(); current = next) {
        next = current->next();
        m_segments.remove(current);
        GCArraySegment<T>::destroy(current);
    }
    m_top = 0;
    m_numberOfSegments = 1;
#if ASSERT_ENABLED
    m_segments.head()->m_top = 0;
#endif
}

template <typename T>
void GCSegmentedArray<T>::expand()
{
    ASSERT(m_segments.head()->m_top == s_segmentCapacity);
    
    GCArraySegment<T>* nextSegment = GCArraySegment<T>::create();
    m_numberOfSegments++;
    
#if ASSERT_ENABLED
    nextSegment->m_top = 0;
#endif

    m_segments.push(nextSegment);
    setTopForEmptySegment();
    validatePrevious();
}

template <typename T>
bool GCSegmentedArray<T>::refill()
{
    validatePrevious();
    if (top())
        return true;
    GCArraySegment<T>::destroy(m_segments.removeHead());
    ASSERT(m_numberOfSegments > 1);
    m_numberOfSegments--;
    setTopForFullSegment();
    validatePrevious();
    return true;
}

template <typename T>
void GCSegmentedArray<T>::fillVector(Vector<T>& vector)
{
    ASSERT(vector.size() == size());

    GCArraySegment<T>* currentSegment = m_segments.head();
    if (!currentSegment)
        return;

    unsigned count = 0;
    for (unsigned i = 0; i < m_top; ++i) {
        ASSERT(currentSegment->data()[i]);
        vector[count++] = currentSegment->data()[i];
    }

    currentSegment = currentSegment->next();
    while (currentSegment) {
        for (unsigned i = 0; i < s_segmentCapacity; ++i) {
            ASSERT(currentSegment->data()[i]);
            vector[count++] = currentSegment->data()[i];
        }
        currentSegment = currentSegment->next();
    }
}

template <typename T>
inline GCArraySegment<T>* GCArraySegment<T>::create()
{
    return new (NotNull, GCSegmentedArrayMalloc::malloc(blockSize)) GCArraySegment<T>();
}

template <typename T>
inline void GCArraySegment<T>::destroy(GCArraySegment* segment)
{
    segment->~GCArraySegment();
    GCSegmentedArrayMalloc::free(segment);
}

template <typename T>
inline size_t GCSegmentedArray<T>::postIncTop()
{
    size_t result = m_top++;
    ASSERT(result == m_segments.head()->m_top++);
    return result;
}

template <typename T>
inline size_t GCSegmentedArray<T>::preDecTop()
{
    size_t result = --m_top;
    ASSERT(result == --m_segments.head()->m_top);
    return result;
}

template <typename T>
inline void GCSegmentedArray<T>::setTopForFullSegment()
{
    ASSERT(m_segments.head()->m_top == s_segmentCapacity);
    m_top = s_segmentCapacity;
}

template <typename T>
inline void GCSegmentedArray<T>::setTopForEmptySegment()
{
    ASSERT(!m_segments.head()->m_top);
    m_top = 0;
}

template <typename T>
inline size_t GCSegmentedArray<T>::top()
{
    ASSERT(m_top == m_segments.head()->m_top);
    return m_top;
}

template <typename T>
#if ASSERT_ENABLED
inline void GCSegmentedArray<T>::validatePrevious()
{
    ASSERT(m_segments.size() == m_numberOfSegments);
}
#else
inline void GCSegmentedArray<T>::validatePrevious() { }
#endif // ASSERT_ENABLED

template <typename T>
inline void GCSegmentedArray<T>::append(T value)
{
    if (m_top == s_segmentCapacity)
        expand();
    m_segments.head()->data()[postIncTop()] = value;
}

template <typename T>
inline bool GCSegmentedArray<T>::canRemoveLast()
{
    return !!m_top;
}

template <typename T>
inline const T GCSegmentedArray<T>::removeLast()
{
    return m_segments.head()->data()[preDecTop()];
}

template <typename T>
inline bool GCSegmentedArray<T>::isEmpty() const
{
    // This happens to be safe to call concurrently. It's important to preserve that capability.
    if (m_top)
        return false;
    if (m_segments.head()->next())
        return false;
    return true;
}

template <typename T>
inline size_t GCSegmentedArray<T>::size() const
{
    return m_top + s_segmentCapacity * (m_numberOfSegments - 1);
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
