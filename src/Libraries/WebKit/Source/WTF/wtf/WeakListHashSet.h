/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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

#include <wtf/ListHashSet.h>
#include <wtf/WeakPtr.h>

namespace WTF {

// Value will be deleted lazily upon rehash or amortized over time. For manual cleanup, call removeNullReferences().
template <typename T, typename WeakPtrImpl, EnableWeakPtrThreadingAssertions assertionsPolicy>
class WeakListHashSet final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    using WeakPtrImplSet = ListHashSet<Ref<WeakPtrImpl>>;
    using AddResult = typename WeakPtrImplSet::AddResult;

    template <typename ListHashSetType, typename IteratorType>
    class WeakListHashSetIteratorBase {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = ptrdiff_t;
        using pointer = T*;
        using reference = T&;

    protected:
        WeakListHashSetIteratorBase(ListHashSetType& set, IteratorType position)
            : m_set { set }
            , m_position { position }
            , m_beginPosition { set.m_set.begin() }
            , m_endPosition { set.m_set.end() }
        {
            skipEmptyBuckets();
        }

        WeakListHashSetIteratorBase(const ListHashSetType& set, IteratorType position)
            : m_set { set }
            , m_position { position }
            , m_beginPosition { set.m_set.begin() }
            , m_endPosition { set.m_set.end() }
        {
            skipEmptyBuckets();
        }

        ~WeakListHashSetIteratorBase() = default;

        ALWAYS_INLINE T* makePeek() { return static_cast<T*>((*m_position)->template get<T>()); }
        ALWAYS_INLINE const T* makePeek() const { return static_cast<const T*>((*m_position)->template get<T>()); }

        void advance()
        {
            ASSERT(m_position != m_endPosition);
            ++m_position;
            skipEmptyBuckets();
            m_set.increaseOperationCountSinceLastCleanup();
        }

        void advanceBackwards()
        {
            ASSERT(m_position != m_beginPosition);
            --m_position;
            while (m_position != m_beginPosition && !makePeek())
                --m_position;
            m_set.increaseOperationCountSinceLastCleanup();
        }

        void skipEmptyBuckets()
        {
            while (m_position != m_endPosition && !makePeek())
                ++m_position;
        }

        const ListHashSetType& m_set;
        IteratorType m_position;
        IteratorType m_beginPosition;
        IteratorType m_endPosition;
    };

    class WeakListHashSetIterator : public WeakListHashSetIteratorBase<WeakListHashSet, typename WeakPtrImplSet::iterator> {
    public:
        using Base = WeakListHashSetIteratorBase<WeakListHashSet, typename WeakPtrImplSet::iterator>;

        bool operator==(const WeakListHashSetIterator& other) const { return Base::m_position == other.Base::m_position; }

        T* get() { return Base::makePeek(); }
        T& operator*() { return *Base::makePeek(); }
        T* operator->() { return Base::makePeek(); }

        WeakListHashSetIterator& operator++()
        {
            Base::advance();
            return *this;
        }

        WeakListHashSetIterator& operator--()
        {
            Base::advanceBackwards();
            return *this;
        }

    private:
        WeakListHashSetIterator(WeakListHashSet& map, typename WeakPtrImplSet::iterator position)
            : Base { map, position }
        { }

        template <typename, typename, EnableWeakPtrThreadingAssertions> friend class WeakListHashSet;
    };

    class WeakListHashSetConstIterator : public WeakListHashSetIteratorBase<WeakListHashSet, typename WeakPtrImplSet::const_iterator> {
    public:
        using Base = WeakListHashSetIteratorBase<WeakListHashSet, typename WeakPtrImplSet::const_iterator>;

        bool operator==(const WeakListHashSetConstIterator& other) const { return Base::m_position == other.Base::m_position; }

        const T* get() const { return Base::makePeek(); }
        const T& operator*() const { return *Base::makePeek(); }
        const T* operator->() const { return Base::makePeek(); }

        WeakListHashSetConstIterator& operator++()
        {
            Base::advance();
            return *this;
        }

        WeakListHashSetConstIterator& operator--()
        {
            Base::advanceBackwards();
            return *this;
        }

    private:
        WeakListHashSetConstIterator(const WeakListHashSet& map, typename WeakPtrImplSet::const_iterator position)
            : Base { map, position }
        { }

        template <typename, typename, EnableWeakPtrThreadingAssertions> friend class WeakListHashSet;
    };

    using iterator = WeakListHashSetIterator;
    using const_iterator = WeakListHashSetConstIterator;

    iterator begin() { return WeakListHashSetIterator(*this, m_set.begin()); }
    iterator end() { return WeakListHashSetIterator(*this, m_set.end()); }
    const_iterator begin() const { return WeakListHashSetConstIterator(*this, m_set.begin()); }
    const_iterator end() const { return WeakListHashSetConstIterator(*this, m_set.end()); }

    template <typename U>
    iterator find(const U& value)
    {
        increaseOperationCountSinceLastCleanup();
        if (auto* impl = value.weakImplIfExists(); impl && *impl)
            return WeakListHashSetIterator(*this, m_set.find(*impl));
        return end();
    }

    template <typename U>
    const_iterator find(const U& value) const
    {
        increaseOperationCountSinceLastCleanup();
        if (auto* impl = value.weakImplIfExists(); impl && *impl)
            return WeakListHashSetConstIterator(*this, m_set.find(*impl));
        return end();
    }

    template <typename U>
    bool contains(const U& value) const
    {
        increaseOperationCountSinceLastCleanup();
        if (auto* impl = value.weakImplIfExists(); impl && *impl)
            return m_set.contains(*impl);
        return false;
    }

    template <typename U>
    AddResult add(const U& value)
    {
        amortizedCleanupIfNeeded();
        return m_set.add(WeakRef<T, WeakPtrImpl>(value).releaseImpl());
    }

    template <typename U>
    AddResult appendOrMoveToLast(const U& value)
    {
        amortizedCleanupIfNeeded();
        return m_set.appendOrMoveToLast(WeakRef<T, WeakPtrImpl>(value).releaseImpl());
    }

    template <typename U>
    bool moveToLastIfPresent(const U& value)
    {
        amortizedCleanupIfNeeded();
        return m_set.moveToLastIfPresent(WeakRef<T, WeakPtrImpl>(value).releaseImpl());
    }

    template <typename U>
    AddResult prependOrMoveToFirst(const U& value)
    {
        amortizedCleanupIfNeeded();
        return m_set.prependOrMoveToFirst(WeakRef<T, WeakPtrImpl>(value).releaseImpl());
    }

    template <typename U, typename V>
    AddResult insertBefore(const U& beforeValue, const V& value)
    {
        amortizedCleanupIfNeeded();
        return m_set.insertBefore(WeakRef<T, WeakPtrImpl>(beforeValue).releaseImpl(), WeakRef<T, WeakPtrImpl>(value).releaseImpl());
    }

    template <typename U>
    AddResult insertBefore(iterator it, const U& value)
    {
        amortizedCleanupIfNeeded();
        return m_set.insertBefore(it.m_position, WeakRef<T, WeakPtrImpl>(value).releaseImpl());
    }

    template <typename U>
    bool remove(const U& value)
    {
        amortizedCleanupIfNeeded();
        if (auto* impl = value.weakImplIfExists(); impl && *impl)
            return m_set.remove(*impl);
        return false;
    }

    bool remove(iterator it)
    {
        if (it == end())
            return false;
        auto result = m_set.remove(it.m_position);
        amortizedCleanupIfNeeded();
        return result;
    }

    void clear()
    {
        m_set.clear();
        cleanupHappened();
    }

    unsigned capacity() const { return m_set.capacity(); }

    bool isEmptyIgnoringNullReferences() const
    {
        if (m_set.isEmpty())
            return true;

        auto onlyContainsNullReferences = begin() == end();
        if (UNLIKELY(onlyContainsNullReferences))
            const_cast<WeakListHashSet&>(*this).clear();
        return onlyContainsNullReferences;
    }

    bool hasNullReferences() const
    {
        unsigned count = 0;
        auto result = WTF::anyOf(m_set, [&] (auto& iterator) {
            ++count;
            return !iterator.get();
        });
        if (result)
            increaseOperationCountSinceLastCleanup(count);
        else
            cleanupHappened();
        return result;
    }

    unsigned computeSize() const
    {
        const_cast<WeakListHashSet&>(*this).removeNullReferences();
        return m_set.size();
    }

    void checkConsistency() { } // To be implemented.

    bool removeNullReferences()
    {
        bool didRemove = false;
        auto it = m_set.begin();
        while (it != m_set.end()) {
            auto currentIt = it;
            ++it;
            if (!currentIt->get()) {
                m_set.remove(currentIt);
                didRemove = true;
            }
        }
        cleanupHappened();
        return didRemove;
    }

    T& first()
    {
        auto it = begin();
        ASSERT(it != end());
        return *it;
    }

    const T& first() const
    {
        auto it = begin();
        ASSERT(it != end());
        return *it;
    }

    T& takeFirst()
    {
        auto it = begin();
        auto& first = *it;
        remove(it);
        return first;
    }

    T* tryTakeFirst()
    {
        auto it = begin();
        if (it == end())
            return nullptr;
        auto* first = it.get();
        remove(it);
        return first;
    }

    T& last()
    {
        auto it = end();
        --it;
        return *it;
    }

    const T& last() const
    {
        auto it = end();
        --it;
        return *it;
    }

private:
    ALWAYS_INLINE void cleanupHappened() const
    {
        m_operationCountSinceLastCleanup = 0;
        m_maxOperationCountWithoutCleanup = std::min(std::numeric_limits<unsigned>::max() / 2, m_set.size()) * 2;
    }

    ALWAYS_INLINE unsigned increaseOperationCountSinceLastCleanup(unsigned count = 1) const
    {
        unsigned currentCount = m_operationCountSinceLastCleanup += count;
        return currentCount;
    }

    ALWAYS_INLINE void amortizedCleanupIfNeeded(unsigned count = 1) const
    {
        unsigned currentCount = increaseOperationCountSinceLastCleanup(count);
        if (currentCount > m_maxOperationCountWithoutCleanup)
            const_cast<WeakListHashSet&>(*this).removeNullReferences();
    }

    WeakPtrImplSet m_set;
    mutable unsigned m_operationCountSinceLastCleanup { 0 };
    mutable unsigned m_maxOperationCountWithoutCleanup { 0 };
};

template<typename T, typename WeakMapImpl>
size_t containerSize(const WeakListHashSet<T, WeakMapImpl>& container) { return container.computeSize(); }

template<typename T, typename WeakMapImpl>
inline auto copyToVector(const WeakListHashSet<T, WeakMapImpl>& collection) -> Vector<WeakPtr<T, WeakMapImpl>> {
    return WTF::map(collection, [](auto& v) -> WeakPtr<T, WeakMapImpl> { return WeakPtr<T, WeakMapImpl> { v }; });
}

}

using WTF::WeakListHashSet;
