/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/Vector.h>

namespace WTF {

template<typename T>
class ThreadSafeWeakHashSet final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    ThreadSafeWeakHashSet() = default;
    ThreadSafeWeakHashSet(ThreadSafeWeakHashSet&& other) { moveFrom(WTFMove(other)); }
    ThreadSafeWeakHashSet& operator=(ThreadSafeWeakHashSet&& other)
    {
        moveFrom(WTFMove(other));
        return *this;
    }

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using pointer = const value_type*;
        using reference = const value_type&;

    private:
        const_iterator(Vector<Ref<T>>&& strongReferences)
            : m_strongReferences(WTFMove(strongReferences)) { }

    public:
        T* get() const
        {
            RELEASE_ASSERT(m_position < m_strongReferences.size());
            return m_strongReferences[m_position].ptr();
        }
        T& operator*() const { return *get(); }
        T* operator->() const { return get(); }

        const_iterator& operator++()
        {
            RELEASE_ASSERT(m_position < m_strongReferences.size());
            ++m_position;
            return *this;
        }

        bool operator==(const const_iterator& other) const
        {
            // This should only be used to compare with end.
            ASSERT_UNUSED(other, other.m_strongReferences.isEmpty());
            return m_position == m_strongReferences.size();
        }

    private:
        template<typename> friend class ThreadSafeWeakHashSet;

        Vector<Ref<T>> m_strongReferences;
        size_t m_position { 0 };
    };

    const_iterator begin() const
    {
        return { values() };
    }

    const_iterator end() const { return { { } }; }

    template<typename U>
    void add(const U& value) requires (std::is_convertible_v<U*, T*>)
    {
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(!value.controlBlock().objectHasStartedDeletion());
        Locker locker { m_lock };
        ControlBlockRefPtr retainedControlBlock { &value.controlBlock() };
        ASSERT(retainedControlBlock);
        amortizedCleanupIfNeeded();
        m_set.add(std::make_pair(WTFMove(retainedControlBlock), &value));
    }

    template<typename U>
    bool remove(const U& value) requires (std::is_convertible_v<U*, T*>)
    {
        Locker locker { m_lock };
        amortizedCleanupIfNeeded();
        // If there are no weak refs then it can't be in our table. In that case
        // there's no point in potentially allocating a ControlBlock.
        if (!value.weakRefCount())
            return false;

        auto it = m_set.find(std::make_pair(&value.controlBlock(), &value));
        if (it == m_set.end())
            return false;
        bool wasDeleted = it->first->objectHasStartedDeletion();
        bool result = m_set.remove(it);
        ASSERT_UNUSED(result, result);
        return !wasDeleted;
    }

    void clear()
    {
        Locker locker { m_lock };
        m_set.clear();
        cleanupHappened();
    }

    template<typename U>
    bool contains(const U& value) const requires (std::is_convertible_v<U*, T*>)
    {
        Locker locker { m_lock };
        amortizedCleanupIfNeeded();
        // If there are no weak refs then it can't be in our table. In that case
        // there's no point in potentially allocating a ControlBlock.
        if (!value.weakRefCount())
            return false;

        auto it = m_set.find(std::make_pair(&value.controlBlock(), &value));
        if (it == m_set.end())
            return false;

        bool wasDeleted = it->first->objectHasStartedDeletion();
        if (wasDeleted)
            m_set.remove(it);
        return !wasDeleted;
    }

    bool isEmptyIgnoringNullReferences() const
    {
        Locker locker { m_lock };
        amortizedCleanupIfNeeded();
        // FIXME: This seems like it should remove any stale entries it finds along the way. Although, it might require a
        // HashSet::removeNoRehash function. https://bugs.webkit.org/show_bug.cgi?id=283928
        for (auto& pair : m_set) {
            if (!pair.first->objectHasStartedDeletion())
                return false;
        }
        return true;
    }

    Vector<Ref<T>> values() const
    {
        Vector<Ref<T>> strongReferences;
        {
            Locker locker { m_lock };
            bool hasNullReferences = false;
            strongReferences = compactMap(m_set, [&hasNullReferences](auto& pair) -> RefPtr<T> {
                if (RefPtr strongReference = pair.first->template makeStrongReferenceIfPossible<T>(pair.second))
                    return strongReference;
                hasNullReferences = true;
                return nullptr;
            });
            if (hasNullReferences)
                m_set.removeIf([](auto& pair) { return pair.first->objectHasStartedDeletion(); });
            cleanupHappened();
        }
        return strongReferences;
    }

    Vector<ThreadSafeWeakPtr<T>> weakValues() const
    {
        Vector<ThreadSafeWeakPtr<T>> weakReferences;
        {
            // FIXME: It seems like this should prune known dead entries as it goes. https://bugs.webkit.org/show_bug.cgi?id=283928
            Locker locker { m_lock };
            weakReferences = WTF::map(m_set, [](auto& pair) {
                return ThreadSafeWeakPtr<T> { *pair.first, *pair.second };
            });
        }
        return weakReferences;
    }

    template<typename Functor>
    void forEach(const Functor& callback) const
    {
        for (auto& item : values())
            callback(item.get());
    }

    unsigned sizeIncludingEmptyEntriesForTesting()
    {
        Locker locker { m_lock };
        return m_set.size();
    }

private:
    ALWAYS_INLINE void cleanupHappened() const WTF_REQUIRES_LOCK(m_lock)
    {
        m_operationCountSinceLastCleanup = 0;
        m_maxOperationCountWithoutCleanup = std::min(std::numeric_limits<unsigned>::max() / 2, m_set.size()) * 2;
    }

    ALWAYS_INLINE void moveFrom(ThreadSafeWeakHashSet&& other)
    {
        Locker locker { m_lock };
        Locker otherLocker { other.m_lock };
        m_set = std::exchange(other.m_set, { });
        m_operationCountSinceLastCleanup = std::exchange(other.m_operationCountSinceLastCleanup, 0);
        m_maxOperationCountWithoutCleanup = std::exchange(other.m_maxOperationCountWithoutCleanup, 0);
    }

    ALWAYS_INLINE void amortizedCleanupIfNeeded() const WTF_REQUIRES_LOCK(m_lock)
    {
        if (++m_operationCountSinceLastCleanup > m_maxOperationCountWithoutCleanup) {
            m_set.removeIf([] (auto& pair) {
                ASSERT(pair.first->weakRefCount());
                return pair.first->objectHasStartedDeletion();
            });
            cleanupHappened();
        }
    }

    mutable UncheckedKeyHashSet<std::pair<ControlBlockRefPtr, const T*>> m_set WTF_GUARDED_BY_LOCK(m_lock);
    mutable unsigned m_operationCountSinceLastCleanup WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    mutable unsigned m_maxOperationCountWithoutCleanup WTF_GUARDED_BY_LOCK(m_lock) { 0 };
    mutable Lock m_lock;
};

} // namespace WTF

using WTF::ThreadSafeWeakHashSet;
