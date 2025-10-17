/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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

#include <wtf/Algorithms.h>
#include <wtf/HashSet.h>
#include <wtf/WeakPtr.h>

namespace WTF {

template<typename T, typename WeakPtrImpl, EnableWeakPtrThreadingAssertions assertionsPolicy>
class WeakHashSet final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    using WeakPtrImplSet = HashSet<Ref<WeakPtrImpl>>;
    using AddResult = typename WeakPtrImplSet::AddResult;

    class WeakHashSetConstIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

    private:
        WeakHashSetConstIterator(const WeakHashSet& set, typename WeakPtrImplSet::const_iterator position)
            : m_set(set)
            , m_position(position)
            , m_endPosition(set.m_set.end())
        {
            skipEmptyBuckets();
        }

    public:
        T* get() const { return static_cast<T*>((*m_position)->template get<T>()); }
        T& operator*() const { return *get(); }
        T* operator->() const { return get(); }

        WeakHashSetConstIterator& operator++()
        {
            ASSERT(m_position != m_endPosition);
            ++m_position;
            skipEmptyBuckets();
            m_set.increaseOperationCountSinceLastCleanup();
            return *this;
        }

        void skipEmptyBuckets()
        {
            while (m_position != m_endPosition && !get())
                ++m_position;
        }

        bool operator==(const WeakHashSetConstIterator& other) const
        {
            return m_position == other.m_position;
        }

    private:
        template <typename, typename, EnableWeakPtrThreadingAssertions> friend class WeakHashSet;

        const WeakHashSet& m_set;
        typename WeakPtrImplSet::const_iterator m_position;
        typename WeakPtrImplSet::const_iterator m_endPosition;
    };
    typedef WeakHashSetConstIterator const_iterator;

    WeakHashSet() { }

    const_iterator begin() const { return WeakHashSetConstIterator(*this, m_set.begin()); }
    const_iterator end() const { return WeakHashSetConstIterator(*this, m_set.end()); }

    template <typename U>
    const_iterator find(const U& value) const
    {
        increaseOperationCountSinceLastCleanup();
        if (auto* impl = value.weakImplIfExists(); impl && *impl)
            return WeakHashSetConstIterator(*this, m_set.find(impl));
        return end();
    }

    template <typename U>
    AddResult add(const U& value)
    {
        amortizedCleanupIfNeeded();
        return m_set.add(WeakRef<T, WeakPtrImpl>(static_cast<const T&>(value)).releaseImpl());
    }

    T* takeAny()
    {
        auto iterator = begin();
        if (iterator == end())
            return nullptr;
        return static_cast<T*>(m_set.take(iterator.m_position)->template get<T>());
    }

    template <typename U>
    bool remove(const U& value)
    {
        amortizedCleanupIfNeeded();
        if (auto* impl = value.weakImplIfExists(); impl && *impl)
            return m_set.remove(*impl);
        return false;
    }

    bool remove(const_iterator iterator)
    {
        bool removed = m_set.remove(iterator.m_position);
        amortizedCleanupIfNeeded();
        return removed;
    }

    void clear()
    {
        m_set.clear();
        cleanupHappened();
    }

    template <typename U>
    bool contains(const U& value) const
    {
        increaseOperationCountSinceLastCleanup();
        if (auto* impl = value.weakImplIfExists(); impl && *impl)
            return m_set.contains(*impl);
        return false;
    }

    unsigned capacity() const { return m_set.capacity(); }

    bool isEmptyIgnoringNullReferences() const
    {
        if (m_set.isEmpty())
            return true;
        return begin() == end();
    }

    bool hasNullReferences() const
    {
        unsigned count = 0;
        auto result = WTF::anyOf(m_set, [&](auto& value) {
            ++count;
            return !value.get();
        });
        if (result)
            increaseOperationCountSinceLastCleanup(count);
        else
            cleanupHappened();
        return result;
    }

    unsigned computeSize() const
    {
        const_cast<WeakHashSet&>(*this).removeNullReferences();
        return m_set.size();
    }

    void forEach(NOESCAPE const Function<void(T&)>& callback)
    {
        auto items = map(m_set, [](const Ref<WeakPtrImpl>& item) {
            auto* pointer = static_cast<T*>(item->template get<T>());
            return WeakPtr<T, WeakPtrImpl> { pointer };
        });
        for (auto& item : items) {
            // FIXME: This contains check is only necessary if the set is being mutated during iteration.
            // Change it to an assertion, or make this function use begin() and end().
            if (item && m_set.contains(*item.m_impl))
                callback(*item);
        }
    }

#if ASSERT_ENABLED
    void checkConsistency() const { m_set.checkConsistency(); }
#else
    void checkConsistency() const { }
#endif

private:
    ALWAYS_INLINE void cleanupHappened() const
    {
        m_operationCountSinceLastCleanup = 0;
        m_maxOperationCountWithoutCleanup = std::min(std::numeric_limits<unsigned>::max() / 2, m_set.size()) * 2;
    }

    ALWAYS_INLINE bool removeNullReferences()
    {
        bool didRemove = m_set.removeIf([] (auto& value) { return !value.get(); });
        cleanupHappened();
        return didRemove;
    }

    ALWAYS_INLINE unsigned increaseOperationCountSinceLastCleanup(unsigned count = 1) const
    {
        unsigned currentCount = m_operationCountSinceLastCleanup += count;
        return currentCount;
    }

    ALWAYS_INLINE void amortizedCleanupIfNeeded() const
    {
        unsigned currentCount = increaseOperationCountSinceLastCleanup();
        if (currentCount > m_maxOperationCountWithoutCleanup)
            const_cast<WeakHashSet&>(*this).removeNullReferences();
    }

    WeakPtrImplSet m_set;
    mutable unsigned m_operationCountSinceLastCleanup { 0 };
    mutable unsigned m_maxOperationCountWithoutCleanup { 0 };
};

template<typename T, typename WeakMapImpl>
size_t containerSize(const WeakHashSet<T, WeakMapImpl>& container) { return container.computeSize(); }

template<typename T, typename WeakMapImpl>
inline auto copyToVector(const WeakHashSet<T, WeakMapImpl>& collection) -> Vector<WeakPtr<T, WeakMapImpl>>
{
    return WTF::map(collection, [](auto& v) -> WeakPtr<T, WeakMapImpl> { return WeakPtr<T, WeakMapImpl> { v }; });
}


} // namespace WTF

using WTF::WeakHashSet;
