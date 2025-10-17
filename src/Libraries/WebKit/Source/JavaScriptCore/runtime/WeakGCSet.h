/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 26, 2024.
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

#include "WeakGCHashTable.h"
#include "WeakInlines.h"
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

// A HashSet with Weak<JSCell> values, which automatically removes values once they're garbage collected.

template<typename T>
struct WeakGCSetHashTraits : HashTraits<Weak<T>> {
    static constexpr bool hasIsReleasedWeakValueFunction = true;
    static bool isReleasedWeakValue(const Weak<T>& value)
    {
        return !value.isHashTableDeletedValue() && !value.isHashTableEmptyValue() && !value;
    }
};

template<typename T>
struct WeakGCSetHash {
    // We only prune stale entries on Full GCs so we have to handle non-Live entries in the table.
    static unsigned hash(const Weak<T>& p) { return PtrHash<T*>::hash(p.get()); }
    static bool equal(const Weak<T>& a, const Weak<T>& b)
    {
        if (!a || !b)
            return false;
        return a.get() == b.get();
    }
    static constexpr bool safeToCompareToEmptyOrDeleted = false;
};

// FIXME: This doesn't currently accept WeakHandleOwners by default... it's probably not hard to add but it's not exactly clear how to handle multiple different handle owners for the same value.
template<typename ValueArg, typename HashArg = WeakGCSetHash<ValueArg>, typename TraitsArg = WeakGCSetHashTraits<ValueArg>>
class WeakGCSet final : public WeakGCHashTable {
    WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE(WeakGCSet);
    using ValueType = Weak<ValueArg>;
    using HashSetType = UncheckedKeyHashSet<ValueType, HashArg, TraitsArg>;

public:
    using AddResult = typename HashSetType::AddResult;
    using iterator = typename HashSetType::iterator;
    using const_iterator = typename HashSetType::const_iterator;

    inline explicit WeakGCSet(VM&);
    inline ~WeakGCSet() final;

    void clear()
    {
        m_set.clear();
    }

    AddResult add(ValueArg* key)
    {
        // Constructing a Weak shouldn't trigger a GC but add this ASSERT for good measure.
        DisallowGC disallowGC;
        return m_set.add(key);
    }

    template<typename HashTranslator, typename T>
    ValueArg* ensureValue(T&& key, const Invocable<ValueType()> auto& functor)
    {
        // If functor invokes GC, GC can prune WeakGCSet, and manipulate HashSet while we are touching it in the ensure function.
        // The functor must not invoke GC.
        DisallowGC disallowGC;
        
        auto result = m_set.template ensure<HashTranslator>(std::forward<T>(key), functor);
        return result.iterator->get();
    }

    // It's not safe to call into the VM or allocate an object while an iterator is open.
    inline iterator begin() { return m_set.begin(); }
    inline const_iterator begin() const { return m_set.begin(); }

    inline iterator end() { return m_set.end(); }
    inline const_iterator end() const { return m_set.end(); }

    // FIXME: Add support for find/contains/remove from a ValueArg* via a HashTranslator.

private:
    void pruneStaleEntries() final;

    HashSetType m_set;
    VM& m_vm;
};

} // namespace JSC
